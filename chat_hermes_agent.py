"""
ChatHermesAgent — 将 Hermes Agent 作为 LangChain BaseChatModel 后端

提供与 ChatOpenAI 兼容的调用接口，底层通过 Hermes AIAgent 执行，
拥有完整的 agent 能力（工具调用、记忆、学习、技能）。

核心能力:
  - 同步/异步生成：_generate / _agenerate
  - 流式输出：_stream / _astream（利用 Hermes stream_callback）
  - 多轮对话：同一实例内复用 AIAgent 和 session_id
  - 工具调用：bind_tools() 将工具描述注入提示词，Hermes 原生执行
  - 结构化输出：with_structured_output() 通过提示词约束输出格式

依赖：hermes-agent >= 0.18.0（或通过 HERMES_HOME 指定本地路径）

用法:
    from chat_hermes_agent import ChatHermesAgent

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    # 单轮
    response = llm.invoke("分析这个项目的结构")

    # 流式
    for chunk in llm.stream("生成一个冒泡排序函数"):
        print(chunk.content, end="", flush=True)

    # 在 LangGraph 中使用
    from langgraph.graph import StateGraph
    graph = StateGraph(AgentState)
    graph.add_node("agent", lambda s: {"messages": [llm.invoke(s["messages"])]})
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
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
from langchain_core.tools import BaseTool, tool
from pydantic import Field, PrivateAttr

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Hermes 路径发现
# ═══════════════════════════════════════════════════════════════

def _ensure_hermes_importable() -> None:
    """确保 Hermes Agent 可被导入。

    优先级：
    1. 已安装的 hermes-agent 包（pip install hermes-agent）
    2. HERMES_HOME 环境变量指定的路径
    3. 项目内 tests/hermes-agent-main/ 目录
    """
    # 尝试直接导入
    try:
        import run_agent  # noqa: F401
        return
    except ImportError:
        pass

    # 通过 HERMES_HOME 环境变量
    hermes_home = os.environ.get("HERMES_HOME", "")
    if hermes_home and os.path.isdir(hermes_home):
        if hermes_home not in sys.path:
            sys.path.insert(0, hermes_home)
        try:
            import run_agent  # noqa: F401
            return
        except ImportError:
            pass

    # 尝试项目内的 tests/hermes-agent-main/
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _bundled = os.path.join(_current_dir, "tests", "hermes-agent-main")
    if os.path.isdir(_bundled):
        if _bundled not in sys.path:
            sys.path.insert(0, _bundled)
        try:
            import run_agent  # noqa: F401
            return
        except ImportError:
            pass

    raise ImportError(
        "无法导入 Hermes Agent。请：\n"
        "  1. pip install hermes-agent\n"
        "  2. 或设置 HERMES_HOME 环境变量指向 Hermes 源码目录\n"
        "  3. 或将 hermes-agent 源码放入 tests/hermes-agent-main/"
    )


# ═══════════════════════════════════════════════════════════════
# 消息格式化
# ═══════════════════════════════════════════════════════════════

def _hermes_format_messages(
    messages: list[BaseMessage],
) -> tuple[str, str | None, list[dict[str, Any]]]:
    """将 LangChain 消息列表转为 Hermes Agent 可接受的格式。

    Hermes 使用 OpenAI-compatible 消息格式：
    [{"role": "system"|"user"|"assistant"|"tool", "content": "..."}]

    Returns:
        (user_message, system_message, conversation_history)
        - user_message: 最后一条用户消息文本（用于 agent.chat()）
        - system_message: 系统提示词（可选）
        - conversation_history: 之前的对话历史（用于 agent.run_conversation()）
    """
    system_content: str | None = None
    history: list[dict[str, Any]] = []
    last_user_msg: str | None = None

    for i, msg in enumerate(messages):
        if isinstance(msg, SystemMessage):
            existing = system_content or ""
            system_content = existing + ("\n" if existing else "") + str(msg.content)

        elif isinstance(msg, HumanMessage):
            content = _serialize_content(msg.content)
            # 如果这是最后一条消息，作为 user_message
            if i == len(messages) - 1:
                last_user_msg = content
            else:
                history.append({"role": "user", "content": content})

        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                entry["content"] = _serialize_content(msg.content)
            else:
                entry["content"] = ""

            # 传递 tool_calls
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name", "unknown"),
                            "arguments": (
                                tc.get("args", {})
                                if isinstance(tc.get("args"), str)
                                else _safe_json_dumps(tc.get("args", {}))
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            history.append(entry)

        elif isinstance(msg, ToolMessage):
            history.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": _serialize_content(msg.content),
            })

    # 如果没有找到最后的用户消息，使用空字符串
    if last_user_msg is None:
        # 尝试从最后一条非系统消息构建
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        if non_system:
            last_msg = non_system[-1]
            last_user_msg = _serialize_content(last_msg.content)
        else:
            last_user_msg = ""

    return last_user_msg, system_content, history


def _serialize_content(content: Any) -> str:
    """序列化消息内容为字符串。

    支持：
    - 纯文本字符串
    - 多模态内容列表（提取 text 部分）
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    text_parts.append("[图片]")
            elif isinstance(block, str):
                text_parts.append(block)
            else:
                text_parts.append(str(block))
        return "\n".join(text_parts)
    return str(content)


def _safe_json_dumps(obj: Any) -> str:
    """安全地序列化对象为 JSON 字符串。"""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(obj)


# ═══════════════════════════════════════════════════════════════
# ChatHermesAgent
# ═══════════════════════════════════════════════════════════════

class ChatHermesAgent(BaseChatModel):
    """Hermes Agent 作为 LangChain ChatModel 后端。

    底层通过 Hermes AIAgent 执行，拥有完整的 agent 能力：
    工具调用、记忆、学习、技能系统。

    Attributes:
        working_dir: 工作目录
        base_url: API 端点地址，默认 "http://localhost:30000/v1"
        api_key: API 密钥（可选，Hermes 可通过 ~/.hermes/config 配置）
        model: 模型名称，默认空字符串（使用 Hermes 配置的默认模型）
        provider: 提供商类型（如 "openai"、"anthropic"），默认自动检测
        max_iterations: 最大工具调用轮次，默认 90
        timeout: 超时秒数，默认 600
        enabled_toolsets: 启用的工具集列表
        disabled_toolsets: 禁用的工具集列表
        system_prompt: 自定义系统提示词
        quiet_mode: 静默模式，默认 True
        reasoning_config: 推理配置（如 max_thinking_tokens）
    """

    # ── 公开配置 ──
    working_dir: str = Field(default=".", description="工作目录")
    base_url: str = Field(
        default="http://localhost:30000/v1",
        description="API 端点地址",
    )
    api_key: str | None = Field(default=None, description="API 密钥")
    model: str = Field(default="", description="模型名称")
    provider: str | None = Field(default=None, description="提供商类型")
    max_iterations: int = Field(
        default=90, ge=1, le=500, description="最大工具调用轮次"
    )
    timeout: int = Field(default=600, ge=10, le=3600, description="超时秒数")
    enabled_toolsets: list[str] | None = Field(
        default=None, description="启用的工具集"
    )
    disabled_toolsets: list[str] | None = Field(
        default=None, description="禁用的工具集"
    )
    system_prompt: str | None = Field(default=None, description="系统提示词")
    quiet_mode: bool = Field(default=True, description="静默模式")
    reasoning_config: dict[str, Any] | None = Field(
        default=None, description="推理配置"
    )

    # ── 内部状态 ──
    _agent: Any = PrivateAttr(default=None)
    _session_id: str = PrivateAttr()
    _bound_tools: list[BaseTool] | None = PrivateAttr(default=None)
    _output_schema_desc: str | None = PrivateAttr(default=None)
    _hermes_initialized: bool = PrivateAttr(default=False)

    # ── BaseChatModel 标识 ──
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "hermes-agent"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "provider": self.provider,
            "working_dir": self.working_dir,
            "session_id": getattr(self, "_session_id", None),
        }

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._session_id = str(uuid.uuid4())

    # ═══════════════════════════════════════════════════════════
    # Hermes AIAgent 延迟初始化
    # ═══════════════════════════════════════════════════════════

    def _get_agent(self) -> Any:
        """获取或创建 Hermes AIAgent 实例。

        Hermes agent 在同一实例内复用，保持会话上下文。
        """
        if self._agent is not None and self._hermes_initialized:
            return self._agent

        _ensure_hermes_importable()
        from run_agent import AIAgent

        agent_kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "quiet_mode": self.quiet_mode,
            "session_id": self._session_id,
        }

        if self.provider:
            agent_kwargs["provider"] = self.provider
        if self.enabled_toolsets is not None:
            agent_kwargs["enabled_toolsets"] = self.enabled_toolsets
        if self.disabled_toolsets is not None:
            agent_kwargs["disabled_toolsets"] = self.disabled_toolsets
        if self.reasoning_config is not None:
            agent_kwargs["reasoning_config"] = self.reasoning_config

        self._agent = AIAgent(**agent_kwargs)
        self._hermes_initialized = True

        logger.debug(
            "Hermes Agent 已初始化: base_url=%s, model=%s, session=%s",
            self.base_url, self.model or "(default)", self._session_id,
        )
        return self._agent

    # ═══════════════════════════════════════════════════════════
    # 公开方法
    # ═══════════════════════════════════════════════════════════

    def reset_session(self) -> str:
        """开始新的会话，下次调用将没有历史上下文。

        Returns:
            新的 session_id
        """
        self._session_id = str(uuid.uuid4())
        self._agent = None
        self._hermes_initialized = False
        return self._session_id

    def bind_tools(
        self,
        tools: Sequence[BaseTool | type | dict[str, Any]],
        **kwargs: Any,
    ) -> "ChatHermesAgent":
        """绑定工具。工具描述将注入系统提示词，Hermes 原生执行工具。

        Hermes Agent 自行管理工具调用循环，中间步骤对 LangChain 不可见，
        最终返回的是包含工具执行结果的文本回复。

        Args:
            tools: 工具列表（BaseTool 实例 / 类型 / dict）
            **kwargs: 额外参数

        Returns:
            新的 ChatHermesAgent 实例（复制配置并绑定工具）
        """
        new_instance = self.model_copy(deep=True)
        new_instance._session_id = str(uuid.uuid4())
        new_instance._agent = None
        new_instance._hermes_initialized = False

        # 解析工具
        resolved: list[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                resolved.append(t)
            elif isinstance(t, dict):
                from langchain_core.tools import StructuredTool
                resolved.append(StructuredTool.from_function(
                    name=t.get("name", "unknown"),
                    description=t.get("description", ""),
                    func=lambda **kw: str(kw),
                    args_schema=t.get("args_schema", None),
                ))
            elif isinstance(t, type) and issubclass(t, BaseTool):
                resolved.append(t())
            else:
                resolved.append(t() if callable(t) else t)

        new_instance._bound_tools = resolved

        # 构建工具描述追加到系统提示
        tool_descriptions = _build_hermes_tool_description(resolved)
        if new_instance.system_prompt:
            new_instance.system_prompt = (
                new_instance.system_prompt
                + "\n\n## 可用工具\n\n"
                + tool_descriptions
            )
        else:
            new_instance.system_prompt = (
                "## 可用工具\n\n"
                + tool_descriptions
                + "\n\n使用上述工具完成任务。"
            )

        return new_instance

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        **kwargs: Any,
    ) -> "ChatHermesAgent":
        """设置结构化输出模式。

        通过系统提示词注入 JSON schema 约束，要求 Hermes Agent
        返回符合格式的 JSON。

        Args:
            schema: JSON Schema (dict) 或 Pydantic 模型类
            **kwargs: 额外参数

        Returns:
            新的 ChatHermesAgent 实例
        """
        new_instance = self.model_copy(deep=True)
        new_instance._session_id = str(uuid.uuid4())
        new_instance._agent = None
        new_instance._hermes_initialized = False

        # 解析 schema
        if isinstance(schema, dict):
            schema_json = _safe_json_dumps(schema)
            schema_desc = f"请以 JSON 格式回复，必须符合以下 JSON Schema:\n```json\n{schema_json}\n```"
        elif hasattr(schema, "model_json_schema"):
            schema_json = _safe_json_dumps(schema.model_json_schema())
            schema_desc = f"请以 JSON 格式回复，必须符合以下 JSON Schema:\n```json\n{schema_json}\n```"
        else:
            schema_desc = f"请以 JSON 格式回复，必须符合以下类型:\n{_safe_json_dumps(schema)}"

        new_instance._output_schema_desc = schema_desc

        # 注入到系统提示词
        if new_instance.system_prompt:
            new_instance.system_prompt = (
                new_instance.system_prompt + "\n\n" + schema_desc
            )
        else:
            new_instance.system_prompt = schema_desc

        return new_instance

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
        user_message, system_msg, history = _hermes_format_messages(messages)

        if not user_message:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=""))]
            )

        agent = self._get_agent()

        try:
            # 注入系统提示词
            effective_system = system_msg
            if self.system_prompt:
                effective_system = (
                    (effective_system + "\n\n" + self.system_prompt)
                    if effective_system
                    else self.system_prompt
                )
            if self._output_schema_desc:
                effective_system = (
                    (effective_system + "\n\n" + self._output_schema_desc)
                    if effective_system
                    else self._output_schema_desc
                )

            # 临时设置系统提示词
            if effective_system:
                agent.ephemeral_system_prompt = effective_system

            # 调用 Hermes agent
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=history if history else None,
            )

            final_response = result.get("final_response", "")

            # 检查 stop words
            if stop:
                for sw in stop:
                    idx = final_response.find(sw)
                    if idx >= 0:
                        final_response = final_response[:idx]
                        break

            return ChatResult(
                generations=[
                    ChatGeneration(message=AIMessage(content=final_response))
                ],
                llm_output={
                    "session_id": self._session_id,
                    "model": self.model or result.get("model", ""),
                },
            )

        except Exception as e:
            logger.error("Hermes Agent 调用失败: %s", e)
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(content=f"[错误] Hermes Agent 调用失败: {e}")
                    )
                ],
                llm_output={"error": str(e)},
            )

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
        user_message, system_msg, history = _hermes_format_messages(messages)

        if not user_message:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return

        agent = self._get_agent()

        # 系统提示词
        effective_system = system_msg
        if self.system_prompt:
            effective_system = (
                (effective_system + "\n\n" + self.system_prompt)
                if effective_system
                else self.system_prompt
            )
        if self._output_schema_desc:
            effective_system = (
                (effective_system + "\n\n" + self._output_schema_desc)
                if effective_system
                else self._output_schema_desc
            )

        if effective_system:
            agent.ephemeral_system_prompt = effective_system

        # 收集的文本用于 stop word 检查
        collected_text: list[str] = []
        stop_triggered: bool = False

        def stream_callback(delta: str) -> None:
            """Hermes stream_callback — 在子线程中调用。"""
            nonlocal stop_triggered
            if stop_triggered:
                return
            collected_text.append(delta)

        try:
            # 发起流式调用（在单独线程中执行）
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    agent.run_conversation,
                    user_message=user_message,
                    conversation_history=history if history else None,
                    stream_callback=stream_callback,
                )

                # 轮询 collected_text，逐块 yield
                last_idx = 0
                while not future.done() or last_idx < len(collected_text):
                    if last_idx < len(collected_text):
                        text = collected_text[last_idx]
                        last_idx += 1

                        # stop word 检查
                        if stop and not stop_triggered:
                            full_text = "".join(collected_text)
                            for sw in stop:
                                idx = full_text.find(sw)
                                if idx >= 0:
                                    stop_triggered = True
                                    break

                        if not stop_triggered:
                            yield ChatGenerationChunk(
                                message=AIMessageChunk(content=text)
                            )

                    elif not future.done():
                        future.result(timeout=0.05)  # 等待或抛异常

                # 确保 future 完成
                future.result(timeout=self.timeout)

        except Exception as e:
            logger.error("Hermes Agent 流式调用失败: %s", e)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=f"[错误] {e}")
            )

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
        """异步生成 — 使用 asyncio.to_thread 包装同步调用。"""
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._generate(messages, stop=stop, **kwargs),
        )

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
        """异步流式 — 在后台线程收集 delta，通过 async 队列 yield。"""
        user_message, system_msg, history = _hermes_format_messages(messages)

        if not user_message:
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            return

        agent = self._get_agent()

        effective_system = system_msg
        if self.system_prompt:
            effective_system = (
                (effective_system + "\n\n" + self.system_prompt)
                if effective_system
                else self.system_prompt
            )
        if self._output_schema_desc:
            effective_system = (
                (effective_system + "\n\n" + self._output_schema_desc)
                if effective_system
                else self._output_schema_desc
            )

        if effective_system:
            agent.ephemeral_system_prompt = effective_system

        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stop_flag = {"triggered": False}

        def stream_callback(delta: str) -> None:
            """在子线程中调用，将 delta 放入 async 队列。"""
            if stop_flag["triggered"]:
                return
            asyncio.run_coroutine_threadsafe(
                queue.put(delta), asyncio.get_event_loop()
            )

        async def _run_and_signal() -> None:
            """在 executor 中运行同步调用，完成后发送结束信号。"""
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: agent.run_conversation(
                        user_message=user_message,
                        conversation_history=history if history else None,
                        stream_callback=stream_callback,
                    ),
                )
            except Exception as e:
                logger.error("Hermes Agent async stream 调用失败: %s", e)
                asyncio.run_coroutine_threadsafe(
                    queue.put(f"[错误] {e}"),
                    loop,
                )
            finally:
                await queue.put(None)  # 结束信号

        runner_task = asyncio.create_task(_run_and_signal())

        collected: list[str] = []
        try:
            while True:
                delta = await queue.get()
                if delta is None:
                    break

                collected.append(delta)

                # stop word 检查
                if stop and not stop_flag["triggered"]:
                    full_text = "".join(collected)
                    for sw in stop:
                        if sw in full_text:
                            stop_flag["triggered"] = True
                            break

                if not stop_flag["triggered"]:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=delta)
                    )
        finally:
            await runner_task


# ═══════════════════════════════════════════════════════════════
# 工具描述构建
# ═══════════════════════════════════════════════════════════════

def _build_hermes_tool_description(tools: list[BaseTool]) -> str:
    """为 Hermes Agent 构建工具描述字符串。"""
    lines: list[str] = []
    for tool in tools:
        desc = getattr(tool, "description", "") or ""
        name = getattr(tool, "name", "unknown")

        # 提取参数信息
        args_info = ""
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            try:
                schema = tool.args_schema.model_json_schema()
                props = schema.get("properties", {})
                if props:
                    params: list[str] = []
                    for pname, pinfo in props.items():
                        ptype = pinfo.get("type", "string")
                        pdesc = pinfo.get("description", "")
                        params.append(f"  - {pname} ({ptype})" +
                                     (f": {pdesc}" if pdesc else ""))
                    args_info = " 参数:\n" + "\n".join(params)
            except Exception:
                pass

        lines.append(f"### {name}\n{desc}")
        if args_info:
            lines.append(args_info)
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 工具形式 — Hermes Agent 作为 LangChain @tool
# ═══════════════════════════════════════════════════════════════

# 全局 Agent 缓存（按 session_id 索引，用于跨工具调用保持会话）
_hermes_agent_cache: dict[str, Any] = {}


def _get_or_create_hermes_agent(
    session_id: str,
    base_url: str = "http://localhost:30000/v1",
    api_key: str | None = None,
    model: str = "",
    max_iterations: int = 90,
    enabled_toolsets: list[str] | None = None,
) -> Any:
    """获取或创建缓存的 Hermes AIAgent 实例。

    按 session_id 缓存，支持跨工具调用保持对话上下文。
    """
    if session_id and session_id in _hermes_agent_cache:
        return _hermes_agent_cache[session_id]

    _ensure_hermes_importable()
    from run_agent import AIAgent

    kwargs: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "max_iterations": max_iterations,
        "quiet_mode": True,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if enabled_toolsets:
        kwargs["enabled_toolsets"] = enabled_toolsets

    agent = AIAgent(**kwargs)

    if session_id:
        _hermes_agent_cache[session_id] = agent

    return agent


def close_hermes_session(session_id: str) -> bool:
    """关闭并清理 Hermes Agent 会话。

    Args:
        session_id: 要关闭的会话 ID

    Returns:
        是否成功关闭
    """
    if session_id in _hermes_agent_cache:
        del _hermes_agent_cache[session_id]
        return True
    return False


def list_hermes_sessions() -> list[str]:
    """列出所有活跃的 Hermes Agent 会话 ID。"""
    return list(_hermes_agent_cache.keys())


@tool
def hermes_agent(
    task: str,
    session_id: str = "",
    base_url: str = "http://localhost:30000/v1",
    api_key: str = "",
    model: str = "",
    max_iterations: int = 90,
    working_dir: str = ".",
) -> str:
    """将任务委托给 Hermes Agent 执行。适合需要以下能力的步骤：
    - 通用 AI Agent 任务（推理、规划、多步骤执行）
    - 利用 Hermes 的记忆和学习能力
    - 文件操作、shell 执行、web 搜索
    - 技能系统（skill）和子 agent 调度

    Args:
        task: 详细的任务描述。越具体越好。
        session_id: 会话 ID。传空字符串每次独立调用；
                    传入相同 ID 则在同一会话内延续（有上下文记忆）。
        base_url: Hermes API 端点地址，默认 http://localhost:30000/v1
        api_key: API 密钥（可选，通常 Hermes 配置已包含）
        model: 模型名称，默认使用 Hermes 配置的默认模型
        max_iterations: 最大工具调用轮次，默认 90
        working_dir: 工作目录（Hermes 当前通过配置管理 working dir）

    Returns:
        Hermes Agent 的执行结果（文本）
    """
    try:
        agent = _get_or_create_hermes_agent(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key or None,
            model=model,
            max_iterations=max_iterations,
        )

        result = agent.run_conversation(user_message=task)
        return result.get("final_response", str(result))

    except Exception as e:
        logger.error("hermes_agent tool 调用失败: %s", e)
        return f"[Hermes Agent 错误] {e}"


@tool
def hermes_agent_structured(
    task: str,
    output_schema: dict,
    session_id: str = "",
    base_url: str = "http://localhost:30000/v1",
    api_key: str = "",
    model: str = "",
    max_iterations: int = 90,
) -> str:
    """将任务委托给 Hermes Agent 并要求结构化 JSON 输出。
    适合需要从上下文中提取结构化信息的步骤。

    Args:
        task: 详细的任务描述。Hermes 会按 output_schema 约束的格式返回。
        output_schema: JSON Schema 定义期望的输出结构。
                       例如: {"type": "object", "properties": {"name": {"type": "string"}}}
        session_id: 会话 ID，同 hermes_agent
        base_url: Hermes API 端点地址
        api_key: API 密钥（可选）
        model: 模型名称
        max_iterations: 最大工具调用轮次

    Returns:
        JSON 字符串（期望符合 output_schema 约束）
    """
    import json

    try:
        agent = _get_or_create_hermes_agent(
            session_id=session_id,
            base_url=base_url,
            api_key=api_key or None,
            model=model,
            max_iterations=max_iterations,
        )

        # 注入 schema 约束到系统提示词
        schema_str = json.dumps(output_schema, ensure_ascii=False)
        structured_prompt = (
            f"{task}\n\n"
            f"请严格按以下 JSON Schema 格式返回结果，只返回 JSON，不要包含其他文字：\n"
            f"```json\n{schema_str}\n```"
        )

        result = agent.run_conversation(user_message=structured_prompt)
        return result.get("final_response", str(result))

    except Exception as e:
        logger.error("hermes_agent_structured tool 调用失败: %s", e)
        return json.dumps({"error": str(e)})


@tool
def hermes_agent_session(
    task: str,
    context: str = "",
    session_id: str = "",
    base_url: str = "http://localhost:30000/v1",
    model: str = "",
    max_iterations: int = 90,
) -> str:
    """在持续会话中将任务委托给 Hermes Agent。适合多步骤工作流，
    每步之间自动保持上下文记忆。

    Args:
        task: 当前步骤的任务描述
        context: 之前步骤的上下文摘要（可选，首次调用可为空）
        session_id: 会话 ID。相同 ID 跨多次调用保持上下文。
        base_url: Hermes API 端点地址
        model: 模型名称
        max_iterations: 最大工具调用轮次

    Returns:
        Hermes Agent 的执行结果（文本）
    """
    try:
        agent = _get_or_create_hermes_agent(
            session_id=session_id,
            base_url=base_url,
            model=model,
            max_iterations=max_iterations,
        )

        # 将 context 合并到任务描述中
        full_task = task
        if context:
            full_task = f"之前的工作上下文：\n{context}\n\n当前任务：\n{task}"

        result = agent.run_conversation(user_message=full_task)
        return result.get("final_response", str(result))

    except Exception as e:
        logger.error("hermes_agent_session tool 调用失败: %s", e)
        return f"[Hermes Agent 会话错误] {e}"


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "ChatHermesAgent",
    "_ensure_hermes_importable",
    "_hermes_format_messages",
    # Tool 形式
    "hermes_agent",
    "hermes_agent_structured",
    "hermes_agent_session",
    "close_hermes_session",
    "list_hermes_sessions",
]
