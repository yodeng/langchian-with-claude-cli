"""
ChatClaudeCode — 将 Claude Code CLI 作为 LangChain BaseChatModel 后端

提供与 ChatAnthropic/ChatOpenAI 兼容的调用接口，底层通过 subprocess 调用 claude CLI。

核心能力：
  - 同步/异步生成：_generate / _agenerate
  - 流式输出：_stream / _astream（解析 stream-json）
  - 多轮对话：自动管理 --session-id / --continue
  - 工具调用：bind_tools() 将工具描述注入提示词，Claude Code 原生执行
  - 结构化输出：with_structured_output() 利用 --json-schema

依赖：claude CLI >= 2.1.162, langchain-core >= 1.0, Python >= 3.10

用法:
    from chat_claude_code import ChatClaudeCode

    llm = ChatClaudeCode(effort="medium", working_dir="/path/to/project")

    # 单轮
    response = llm.invoke("分析这个项目的结构")

    # 多轮（自动在同一 session 内延续）
    response = llm.invoke("列出所有 Python 文件")
    response = llm.invoke("分析第一个文件的功能")  # 知道上下文

    # 流式
    for chunk in llm.stream("生成一个冒泡排序函数"):
        print(chunk.content, end="", flush=True)

    # 在 LangGraph 中使用
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda s: {"messages": [llm.invoke(s["messages"])]})
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import uuid
from typing import Any, AsyncIterator, Iterator, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 消息格式化
# ═══════════════════════════════════════════════════════════════

def _format_messages(messages: list[BaseMessage]) -> tuple[str, str | None]:
    """
    将 LangChain 消息列表转为 Claude Code CLI 可接受的 prompt。

    Returns:
        (conversation_prompt, system_prompt_or_none)
    """
    system_content: str | None = None
    conversation_parts: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            existing = system_content or ""
            system_content = existing + ("\n" if existing else "") + str(msg.content)

        elif isinstance(msg, HumanMessage):
            name_tag = f" (as {msg.name})" if msg.name else ""
            conversation_parts.append(f"Human{name_tag}: {msg.content}")

        elif isinstance(msg, AIMessage):
            # 处理 tool_calls
            content_parts: list[str] = []
            if msg.content:
                content_parts.append(str(msg.content))

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("args", {})
                    if isinstance(tc_args, dict):
                        args_str = json.dumps(tc_args, ensure_ascii=False)
                    else:
                        args_str = str(tc_args)
                    content_parts.append(
                        f"\n[调用工具: {tc_name}({args_str})]"
                    )

            conversation_parts.append(
                f"Assistant: {' '.join(content_parts)}"
            )

        elif isinstance(msg, ToolMessage):
            tool_name = msg.name or "tool"
            conversation_parts.append(
                f"Tool result ({tool_name}): {msg.content}"
            )

        else:
            # 未知类型，尽力处理
            content = getattr(msg, "content", str(msg))
            role = getattr(msg, "type", "unknown")
            conversation_parts.append(f"{role}: {content}")

    conversation = "\n\n".join(conversation_parts)
    return conversation, system_content


# ═══════════════════════════════════════════════════════════════
# Claude Code CLI 响应解析
# ═══════════════════════════════════════════════════════════════

def _parse_json_output(stdout: str) -> dict[str, Any]:
    """解析 claude --output-format json 的输出"""
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"result": stdout}


def _extract_text_from_response(parsed: dict[str, Any]) -> str:
    """从解析后的 JSON 响应中提取文本内容"""
    # 优先从 result 字段取
    result = parsed.get("result", parsed.get("content", ""))

    if isinstance(result, str):
        return result

    if isinstance(result, list):
        texts: list[str] = []
        for block in result:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    texts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    texts.append(
                        f"[Tool: {block.get('name', 'unknown')}"
                        f"({json.dumps(block.get('input', {}), ensure_ascii=False)})]"
                    )
                elif block_type == "tool_result":
                    texts.append(f"[Tool Result: {block.get('content', '')}]")
                else:
                    texts.append(block.get("text", str(block)))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)

    return str(result)


def _extract_usage(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """从 JSON 响应中提取 usage 信息"""
    usage = parsed.get("usage")
    if usage:
        return dict(usage)
    return None


# ═══════════════════════════════════════════════════════════════
# 流式事件解析（同步/异步共享）
# ═══════════════════════════════════════════════════════════════

def _parse_json_line(line: str) -> dict[str, Any] | None:
    """解析单行 JSON，失败返回 None"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _parse_stream_event_line(line: str) -> dict[str, Any] | None:
    """解析一行 stream-json 输出，返回标准化的事件 dict 或 None。

    返回格式: {"type": "text"|"usage"|"stop", ...}
    - text: {"type": "text", "content": "增量文本"}
    - usage: {"type": "usage", "input_tokens": N, "output_tokens": N, "total_tokens": N}
    - stop: {"type": "stop"}
    """
    outer = _parse_json_line(line)
    if outer is None:
        return None

    outer_type = outer.get("type", "")
    if outer_type == "system":
        return None

    # 从 stream_event 包装中提取内层 event
    if outer_type == "stream_event":
        inner = outer.get("event", {})
    else:
        inner = outer

    event_type = inner.get("type", "")

    if event_type == "content_block_delta":
        delta = inner.get("delta", {})
        if isinstance(delta, dict):
            delta_type = delta.get("type", "")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    return {"type": "text", "content": text}
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                if partial:
                    return {"type": "text", "content": partial}
    elif event_type == "message_delta":
        usage = inner.get("usage", {})
        if usage:
            input_t = usage.get("input_tokens", 0)
            output_t = usage.get("output_tokens", 0)
            return {
                "type": "usage",
                "input_tokens": input_t,
                "output_tokens": output_t,
                "total_tokens": input_t + output_t,
            }
    elif event_type == "message_stop":
        return {"type": "stop"}

    return None


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode — 核心类
# ═══════════════════════════════════════════════════════════════

class ChatClaudeCode(BaseChatModel):
    """
    Claude Code CLI 作为 LangChain ChatModel 后端。

    底层通过 `claude -p` 调用，拥有完整的文件系统、shell、git 访问能力。
    适合需要代码分析、文件操作、项目级理解的场景。

    Attributes:
        working_dir: 工作目录，Claude Code 将在此目录下执行
        effort: 努力级别 (low/medium/high/xhigh/max)，默认 medium
        allowed_tools: 允许的工具列表，如 ["Read", "Write", "Bash(git *)"]
        disallowed_tools: 禁止的工具列表
        system_prompt: 自定义系统提示词
        timeout: 超时秒数，默认 300
        skip_permissions: 跳过权限确认，默认 True（自动化场景）
        context_files: 额外授权访问的文件/目录
        extra_env: 额外的环境变量
        model: 模型名称（可选，默认使用 Claude Code 配置的模型）
        max_tokens: 最大输出 token 数（可选）
    """

    # ── 公开配置 ──
    working_dir: str = Field(default=".", description="工作目录")
    effort: str = Field(default="medium", description="努力级别: low/medium/high/xhigh/max")
    allowed_tools: list[str] | None = Field(default=None, description="允许的工具列表")
    disallowed_tools: list[str] | None = Field(default=None, description="禁止的工具列表")
    system_prompt: str | None = Field(default=None, description="系统提示词")
    timeout: int = Field(default=300, ge=10, le=3600, description="超时秒数")
    skip_permissions: bool = Field(default=True, description="跳过权限确认")
    context_files: list[str] | None = Field(default=None, description="授权访问的文件/目录")
    extra_env: dict[str, str] = Field(default_factory=dict, description="额外环境变量")
    model: str | None = Field(default=None, description="模型名称")
    max_tokens: int | None = Field(default=None, description="最大输出 token 数")

    # ── 内部状态 ──
    _session_id: str = PrivateAttr()
    _session_turn: int = PrivateAttr(default=0)
    _bound_tools: list[BaseTool] | None = PrivateAttr(default=None)
    _output_schema: dict[str, Any] | None = PrivateAttr(default=None)

    # ── BaseChatModel 标识 ──
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "working_dir": self.working_dir,
            "effort": self.effort,
            "model": self.model,
            "session_id": getattr(self, "_session_id", None),
        }

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._session_id = str(uuid.uuid4())

    # ═══════════════════════════════════════════════════════════
    # 公开方法
    # ═══════════════════════════════════════════════════════════

    def reset_session(self) -> str:
        """开始新的会话（新的 session_id），下次调用将没有历史上下文。"""
        self._session_id = str(uuid.uuid4())
        self._session_turn = 0
        return self._session_id

    def bind_tools(
        self,
        tools: Sequence[BaseTool | type | dict[str, Any]],
        **kwargs: Any,
    ) -> "ChatClaudeCode":
        """
        绑定工具。工具描述将注入系统提示词，Claude Code 原生执行工具。

        注意：Claude Code CLI 自行管理工具执行，中间步骤对 LangChain 不可见，
        最终返回的是包含工具执行结果的文本回复。

        Args:
            tools: 工具列表（BaseTool 实例 / 类型 / dict）
            **kwargs: 额外参数

        Returns:
            新的 ChatClaudeCode 实例（复制配置并绑定工具）
        """
        # 使用 model_copy 避免修改当前实例
        new_instance = self.model_copy(deep=True)
        new_instance._session_id = str(uuid.uuid4())
        new_instance._session_turn = 0

        # 解析工具
        resolved: list[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                resolved.append(t)
            elif isinstance(t, dict):
                # dict 格式的工具定义，转为 BaseTool
                resolved.append(_dict_to_tool(t))
            elif isinstance(t, type) and issubclass(t, BaseTool):
                resolved.append(t())
            else:
                # 尝试作为可调用对象
                resolved.append(_callable_to_tool(t))

        new_instance._bound_tools = resolved

        # 构建工具描述追加到系统提示
        tool_descriptions = _build_tool_description(resolved)

        # 合并到系统提示
        if new_instance.system_prompt:
            new_instance.system_prompt = (
                new_instance.system_prompt
                + "\n\n## 可用工具\n\n"
                + tool_descriptions
                + "\n\n使用上述工具完成任务。在每次回复前先评估是否需要调用工具。"
            )
        else:
            new_instance.system_prompt = (
                "## 可用工具\n\n"
                + tool_descriptions
                + "\n\n使用上述工具完成任务。在每次回复前先评估是否需要调用工具。"
            )

        # 添加工具到 allowed_tools
        tool_names = [t.name for t in resolved]
        if new_instance.allowed_tools:
            new_instance.allowed_tools = list(
                set(new_instance.allowed_tools + tool_names)
            )
        else:
            # 不强制设置 allowed_tools，让 Claude Code 自行管理权限
            pass

        return new_instance

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        **kwargs: Any,
    ) -> "ChatClaudeCode":
        """
        设置结构化输出模式。使用 --json-schema 约束 Claude Code 的输出格式。

        Args:
            schema: JSON Schema dict 或 Pydantic 模型类
            **kwargs: 额外参数

        Returns:
            新的 ChatClaudeCode 实例
        """
        new_instance = self.model_copy(deep=True)
        new_instance._session_id = str(uuid.uuid4())
        new_instance._session_turn = 0

        # 解析 schema
        if isinstance(schema, dict):
            new_instance._output_schema = schema
        elif isinstance(schema, type):
            # Pydantic 模型 → JSON Schema
            new_instance._output_schema = schema.model_json_schema()
        else:
            raise TypeError(f"schema 必须是 dict 或 Pydantic 模型类，收到: {type(schema)}")

        return new_instance

    # ═══════════════════════════════════════════════════════════
    # 底层命令构建
    # ═══════════════════════════════════════════════════════════

    def _build_command(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> list[str]:
        """构建 claude CLI 命令参数列表"""
        cmd: list[str] = ["claude", "-p", prompt]

        # 输出格式
        if stream:
            cmd.extend([
                "--output-format", "stream-json",
                "--include-partial-messages",
                "--verbose",  # stream-json 要求 --verbose
            ])
        elif self._output_schema:
            cmd.extend(["--output-format", "json"])
            # --json-schema 支持 JSON 字符串
            schema_str = json.dumps(self._output_schema)
            cmd.extend(["--json-schema", schema_str])
        else:
            cmd.extend(["--output-format", "json"])

        # 会话管理
        if self._session_turn == 0:
            cmd.extend(["--session-id", self._session_id])
        else:
            cmd.extend(["--resume", self._session_id])

        # 努力级别
        cmd.extend(["--effort", self.effort])

        # 模型
        if self.model:
            cmd.extend(["--model", self.model])

        # 最大 token
        if self.max_tokens:
            cmd.extend(["--max-tokens", str(self.max_tokens)])

        # 权限
        if self.skip_permissions:
            cmd.append("--dangerously-skip-permissions")

        # 工具限制
        if self.allowed_tools:
            cmd.append("--allowedTools")
            cmd.extend(self.allowed_tools)

        if self.disallowed_tools:
            cmd.append("--disallowedTools")
            cmd.extend(self.disallowed_tools)

        # 上下文文件
        if self.context_files:
            cmd.append("--add-dir")
            cmd.extend(self.context_files)

        # 系统提示
        if system_prompt or self.system_prompt:
            sp = system_prompt or self.system_prompt
            # 如果绑定了工具，系统提示中包含工具描述
            cmd.extend(["--system-prompt", sp])

        return cmd

    def _build_env(self) -> dict[str, str]:
        """构建环境变量"""
        env = os.environ.copy()
        if self.extra_env:
            env.update(self.extra_env)
        return env

    # ═══════════════════════════════════════════════════════════
    # 同步生成
    # ═══════════════════════════════════════════════════════════

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        conversation, system_prompt = _format_messages(messages)

        if not conversation:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        cmd = self._build_command(conversation, system_prompt=system_prompt, stream=False)
        env = self._build_env()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.working_dir,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"[错误] 调用超时（{self.timeout}秒）")
                    )
                ],
                llm_output={"error": "timeout"},
            )
        except FileNotFoundError:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="[错误] claude CLI 未安装或不在 PATH 中")
                    )
                ],
                llm_output={"error": "cli_not_found"},
            )
        except OSError as e:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"[错误] 无法启动 claude CLI: {e}")
                    )
                ],
                llm_output={"error": "os_error", "detail": str(e)},
            )

        self._session_turn += 1

        result_chat = _build_chat_result(
            result.stdout, result.stderr, result.returncode
        )
        assert result_chat is not None
        return result_chat

    # ═══════════════════════════════════════════════════════════
    # 同步流式
    # ═══════════════════════════════════════════════════════════

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        conversation, system_prompt = _format_messages(messages)

        if not conversation:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return

        cmd = self._build_command(conversation, system_prompt=system_prompt, stream=True)
        env = self._build_env()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.working_dir,
                env=env,
            )
        except FileNotFoundError:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="[错误] claude CLI 未安装或不在 PATH 中")
            )
            return
        except OSError as e:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=f"[错误] 无法启动 claude CLI: {e}")
            )
            return

        self._session_turn += 1

        # 逐行读取 stream-json，通过共享解析器处理
        collected_text: list[str] = []
        stop_triggered: bool = False
        try:
            for line in proc.stdout:
                if stop and not stop_triggered:
                    if _check_stop_words(collected_text, stop):
                        proc.terminate()
                        stop_triggered = True
                if stop_triggered:
                    break

                event = _parse_stream_event_line(line.strip())
                if event is None:
                    continue

                chunk = _stream_event_to_chunk(event)
                if chunk is not None:
                    if event["type"] == "text":
                        collected_text.append(event["content"])
                    yield chunk
                # "stop" 事件自然结束，无需处理

        finally:
            # 确保进程终止，防止僵尸进程
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("claude CLI 进程未在 5s 内终止，强制 kill")
                proc.kill()
            except Exception:
                logger.debug("进程终止时忽略异常（进程可能已退出）")

    # ═══════════════════════════════════════════════════════════
    # 异步生成
    # ═══════════════════════════════════════════════════════════

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        conversation, system_prompt = _format_messages(messages)

        if not conversation:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        cmd = self._build_command(conversation, system_prompt=system_prompt, stream=False)
        env = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                pass
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"[错误] 调用超时（{self.timeout}秒）")
                    )
                ],
                llm_output={"error": "timeout"},
            )
        except FileNotFoundError:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content="[错误] claude CLI 未安装或不在 PATH 中")
                    )
                ],
                llm_output={"error": "cli_not_found"},
            )
        except OSError as e:
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"[错误] 无法启动 claude CLI: {e}")
                    )
                ],
                llm_output={"error": "os_error", "detail": str(e)},
            )

        self._session_turn += 1

        stdout_text = stdout.decode("utf-8") if stdout else ""
        stderr_text = stderr.decode("utf-8") if stderr else ""

        result_chat = _build_chat_result(stdout_text, stderr_text, proc.returncode)
        assert result_chat is not None
        return result_chat

    # ═══════════════════════════════════════════════════════════
    # 异步流式
    # ═══════════════════════════════════════════════════════════

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        conversation, system_prompt = _format_messages(messages)

        if not conversation:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return

        cmd = self._build_command(conversation, system_prompt=system_prompt, stream=True)
        env = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
                env=env,
            )
        except FileNotFoundError:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="[错误] claude CLI 未安装或不在 PATH 中")
            )
            return
        except OSError as e:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=f"[错误] 无法启动 claude CLI: {e}")
            )
            return

        self._session_turn += 1

        collected_text: list[str] = []
        stop_triggered: bool = False
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break

                line_text = line.decode("utf-8").strip()
                if not line_text:
                    continue

                # 检查 stop
                if stop and not stop_triggered:
                    if _check_stop_words(collected_text, stop):
                        proc.terminate()
                        stop_triggered = True
                if stop_triggered:
                    break

                event = _parse_stream_event_line(line_text)
                if event is None:
                    continue

                chunk = _stream_event_to_chunk(event)
                if chunk is not None:
                    if event["type"] == "text":
                        collected_text.append(event["content"])
                    yield chunk
                # "stop" 事件自然结束，无需处理
        finally:
            # 确保进程终止，防止僵尸进程
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning("claude CLI 异步进程未在 5s 内终止，强制 kill")
                try:
                    proc.kill()
                except Exception:
                    pass
            except Exception:
                logger.debug("异步进程终止时忽略异常（进程可能已退出）")


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode 共享辅助方法
# ═══════════════════════════════════════════════════════════════


def _check_stop_words(collected_text: list[str], stop_words: list[str]) -> bool:
    """检查已收集文本是否包含停止词"""
    full = "".join(collected_text)
    return any(w in full for w in stop_words)


def _stream_event_to_chunk(
    event: dict[str, Any],
) -> ChatGenerationChunk | None:
    """将流式事件转换为 ChatGenerationChunk"""
    if event["type"] == "text":
        return ChatGenerationChunk(
            message=AIMessageChunk(content=event["content"])
        )
    elif event["type"] == "usage":
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": event["input_tokens"],
                    "output_tokens": event["output_tokens"],
                    "total_tokens": event["total_tokens"],
                },
            )
        )
    return None


def _build_chat_result(stdout_text: str, stderr_text: str, returncode: int) -> ChatResult | None:
    """从子进程输出构建 ChatResult。成功时返回 ChatResult，失败时返回 None 表示错误已由调用方处理。"""
    if returncode != 0:
        error_msg = stderr_text or stdout_text or "未知错误"
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=f"[Claude Code 错误] returncode={returncode}\n{error_msg}"
                    )
                )
            ],
            llm_output={"error": error_msg, "returncode": returncode},
        )

    parsed = _parse_json_output(stdout_text)
    text = _extract_text_from_response(parsed)
    usage = _extract_usage(parsed)

    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=text))],
        llm_output={"usage": usage} if usage else {},
    )


# ═══════════════════════════════════════════════════════════════
# 工具辅助函数
# ═══════════════════════════════════════════════════════════════

def _build_tool_description(tools: list[BaseTool]) -> str:
    """构建工具描述文本，注入到系统提示中"""
    lines: list[str] = []
    for tool in tools:
        desc = tool.description or "无描述"
        # 尝试获取参数 schema
        args_schema = ""
        if hasattr(tool, "args_schema") and tool.args_schema:
            try:
                schema = tool.args_schema.model_json_schema()
                props = schema.get("properties", {})
                if props:
                    args_parts = []
                    for pname, pinfo in props.items():
                        ptype = pinfo.get("type", "any")
                        pdesc = pinfo.get("description", "")
                        required = pname in schema.get("required", [])
                        req_mark = " (必填)" if required else ""
                        args_parts.append(f"    - {pname}: {ptype}{req_mark} — {pdesc}")
                    if args_parts:
                        args_schema = "\n  参数:\n" + "\n".join(args_parts)
            except Exception:
                logger.debug("解析工具参数 schema 失败，跳过参数展示")

        lines.append(f"- **{tool.name}**: {desc}{args_schema}")

    return "\n".join(lines)


def _dict_to_tool(tool_dict: dict[str, Any]) -> BaseTool:
    """将 dict 转为 BaseTool"""
    from langchain_core.tools import StructuredTool

    name = tool_dict.get("name", "unknown_tool")
    description = tool_dict.get("description", "")
    func = tool_dict.get("func", lambda **kwargs: "done")

    # 尝试获取 args_schema
    args_schema = tool_dict.get("args_schema")
    if args_schema is None:
        # 从 function 参数构造
        import inspect

        sig = inspect.signature(func)
        if len(sig.parameters) > 0:
            from pydantic import create_model

            fields = {}
            for pname, param in sig.parameters.items():
                if pname in ("self", "cls"):
                    continue
                annotation = (
                    param.annotation if param.annotation is not inspect.Parameter.empty else str
                )
                default = (
                    param.default
                    if param.default is not inspect.Parameter.empty
                    else ...
                )
                fields[pname] = (annotation, default)
            if fields:
                args_schema = create_model(f"{name}_args", **fields)

    return StructuredTool(
        name=name,
        description=description,
        func=func,
        args_schema=args_schema,
    )


def _callable_to_tool(func: Any) -> BaseTool:
    """将可调用对象转为 BaseTool"""
    from langchain_core.tools import tool as tool_decorator

    return tool_decorator(func)
