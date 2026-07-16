"""
Hermes Agent 集成 — 将 Hermes Agent 编排能力与 Claude Code CLI 执行能力结合

提供 create_hermes_agent() 函数，创建以 Hermes Agent 为 LLM 后端的
LangGraph Agent，并可选择性集成 Claude Code 工具进行代码级任务执行。

用法:
    from hermes_agent import create_hermes_agent

    # 模式 A：Hermes Agent 作为 LLM + Claude Code 工具执行
    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="code_analysis",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "分析项目架构"}]})

    # 模式 B：纯 Hermes Agent（不注册 Claude Code 工具）
    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="none",
    )

    # 模式 C：传入已有的 BaseChatModel 实例
    from chat_hermes_agent import ChatHermesAgent
    llm = ChatHermesAgent(base_url="http://localhost:30000/v1")
    agent = create_hermes_agent(model=llm, mode="full_access")

预设模式:
    - code_analysis: 只读分析（默认）
    - code_refactor: 读写代码
    - full_access:  无限制访问
    - none:         纯 Hermes Agent，不注册 Claude Code 工具

与 deep_agent.py 的区别:
    - deep_agent.py 使用 deepagents.create_deep_agent() 编排框架
    - hermes_agent.py 使用 Hermes AIAgent 作为 LLM 后端（通过 ChatHermesAgent）
    - 两者都支持 Claude Code 工具集成

依赖:
    hermes-agent >= 0.18.0（或 HERMES_HOME 环境变量）
    deepagents >= 0.5.0（Agent 编排方式）
    langchain-core >= 1.0
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 预设配置
# ═══════════════════════════════════════════════════════════════

_PRESETS: dict[str, dict[str, Any]] = {
    "code_analysis": {
        "description": "只读代码分析 — 不允许修改文件",
        "register": ["claude_code", "claude_code_structured"],
        "claude_code_config": {
            "allowed_tools": [
                "Read",
                "Glob",
                "Grep",
                "Bash(ls *)",
                "Bash(find *)",
                "Bash(cat *)",
                "Bash(wc *)",
            ],
            "effort": "low",
        },
        "claude_code_structured_config": {
            "effort": "medium",
        },
    },
    "code_refactor": {
        "description": "代码重构 — 读写文件 + git 操作",
        "register": ["claude_code", "claude_code_structured"],
        "claude_code_config": {
            "allowed_tools": [
                "Read",
                "Write",
                "Edit",
                "Glob",
                "Grep",
                "Bash(ls *)",
                "Bash(find *)",
                "Bash(git *)",
                "Bash(python *)",
            ],
            "effort": "medium",
        },
        "claude_code_structured_config": {
            "effort": "medium",
        },
    },
    "full_access": {
        "description": "完全访问 — 无工具限制，适合复杂任务",
        "register": [
            "claude_code",
            "claude_code_structured",
            "claude_code_isolated",
        ],
        "claude_code_config": {
            "allowed_tools": None,  # None = 不限制
            "effort": "high",
        },
        "claude_code_structured_config": {
            "effort": "high",
        },
        "claude_code_isolated_config": {
            "effort": "high",
        },
    },
    "none": {
        "description": "不注册 Claude Code 工具 — 纯 Hermes Agent",
        "register": [],
    },
}


def _make_preset_tool_list(
    mode: str,
    effort_override: str | None = None,
    allowed_override: list[str] | None = None,
    working_dir: str = ".",
) -> list[Any]:
    """根据预设模式构建 Claude Code 工具列表。

    复用 deep_agent.py 中的工具逻辑。
    """
    from claude_code_tool import claude_code, claude_code_isolated, claude_code_structured
    from langchain_core.tools import BaseTool as _BaseTool

    preset = _PRESETS[mode]
    tools: list[_BaseTool] = []

    for tool_name in preset["register"]:
        if tool_name == "claude_code":
            t = _copy_tool(claude_code)
            config = dict(preset.get("claude_code_config", {}))
            config["working_dir"] = working_dir
            if effort_override is not None:
                config["effort"] = effort_override
            if allowed_override is not None:
                config["allowed_tools"] = allowed_override
            if config:
                t = _apply_tool_defaults(t, config)
            tools.append(t)

        elif tool_name == "claude_code_structured":
            t = _copy_tool(claude_code_structured)
            config = dict(preset.get("claude_code_structured_config", {}))
            config["working_dir"] = working_dir
            if effort_override is not None:
                config["effort"] = effort_override
            if config:
                t = _apply_tool_defaults(t, config)
            tools.append(t)

        elif tool_name == "claude_code_isolated":
            t = _copy_tool(claude_code_isolated)
            config = dict(preset.get("claude_code_isolated_config", {}))
            config["working_dir"] = working_dir
            if effort_override is not None:
                config["effort"] = effort_override
            if config:
                t = _apply_tool_defaults(t, config)
            tools.append(t)

    return tools


def _copy_tool(t: Any) -> Any:
    """浅拷贝工具（通过重新构造），避免修改原始模板。"""
    return t.model_copy(deep=False) if hasattr(t, "model_copy") else t


def _apply_tool_defaults(tool: Any, defaults: dict[str, Any]) -> Any:
    """为工具参数设置默认值，使其在 Agent 调用时无需手动传递。"""
    from langchain_core.tools import StructuredTool

    original = tool.func

    def _wrapper(**kwargs: Any) -> str:
        """带默认值的工具调用包装。"""
        merged: dict[str, Any] = dict(defaults)
        merged.update(kwargs)
        return original(**merged)

    _wrapper.__name__ = original.__name__
    _wrapper.__qualname__ = original.__qualname__
    _wrapper.__doc__ = tool.description
    _wrapper.__annotations__ = {"return": str}

    new_schema = None
    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            from pydantic import create_model

            schema = tool.args_schema.model_json_schema()
            fields: dict[str, Any] = {}
            for name_, info in schema.get("properties", {}).items():
                is_required = name_ in schema.get("required", [])
                if name_ in defaults:
                    fields[name_] = (str, defaults[name_])
                elif is_required:
                    fields[name_] = (str, ...)
                else:
                    fields[name_] = (str, None)

            if fields:
                new_schema = create_model(
                    f"{tool.name}_args_defaults",
                    **fields,
                )
        except Exception:
            logger.debug("更新 args_schema 失败，使用原始 schema")

    return StructuredTool.from_function(
        func=_wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=new_schema or tool.args_schema,
        return_direct=False,
    )


# ═══════════════════════════════════════════════════════════════
# 系统提示
# ═══════════════════════════════════════════════════════════════

def _build_system_prompt(mode: str, custom_prompt: str | None = None) -> str | None:
    """根据模式生成系统提示。"""
    if custom_prompt:
        return custom_prompt

    if mode == "none":
        return None

    preset: dict[str, Any] = _PRESETS[mode]
    description: str = preset["description"]
    registered: list[str] = preset.get("register", [])

    tool_hints = _tool_hints_for_mode(mode)

    return f"""你是一个代码 Agent，由 Hermes Agent 框架提供 LLM 能力，底层通过 Claude Code CLI 执行代码级任务。

**当前模式**：{mode}（{description}）

**可用工具说明**：

- Hermes 内置工具（文件操作、shell 执行、浏览器、web 搜索等）用于通用任务。
{chr(10).join(f'- 已注册 `{t}` 工具：{tool_hints.get(t, "委托代码任务给 Claude Code CLI")}' for t in registered)}

**工作原则**：

1. **先规划再执行**：分析任务，制定计划，然后逐步完成。
2. **代码操作委托 Claude Code**：涉及多文件读取、代码分析、重构、git 操作时，优先用 `claude_code` 工具。Claude Code 拥有完整的项目上下文和原生 shell 能力。
3. **简单操作用 Hermes 内置工具**：单文件读取、简单搜索用内置工具更快。
4. **结构化提取用 `claude_code_structured`**：需要从代码中提取结构化信息时使用。
5. **每次完成后向用户报告进展**，说明已完成的工作和下一步计划。
"""


def _tool_hints_for_mode(mode: str) -> dict[str, str]:
    """返回当前模式下各工具的使用提示。"""
    hints: dict[str, str] = {}
    registered = _PRESETS[mode]["register"]

    if "claude_code" in registered:
        hints["claude_code"] = (
            "委托复杂的代码任务（多文件读写、shell 命令、git 操作、代码分析）。"
            "支持 session_id 保持多步上下文。"
        )
    if "claude_code_structured" in registered:
        hints["claude_code_structured"] = (
            "委托任务并要求结构化 JSON 输出。适合提取函数签名、类定义、依赖关系等。"
        )
    if "claude_code_isolated" in registered:
        hints["claude_code_isolated"] = (
            "在隔离的 git worktree 中执行任务。适合并行处理互不干扰的子任务。"
        )

    return hints


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def create_hermes_agent(
    model: str | BaseChatModel | None = None,
    *,
    tools: Sequence[Any] | None = None,
    mode: str = "code_analysis",
    system_prompt: str | None = None,
    working_dir: str = ".",
    # Hermes 特定配置
    base_url: str = "http://localhost:30000/v1",
    api_key: str | None = None,
    hermes_model: str = "",
    provider: str | None = None,
    max_iterations: int = 90,
    enabled_toolsets: list[str] | None = None,
    disabled_toolsets: list[str] | None = None,
    reasoning_config: dict[str, Any] | None = None,
    # Claude Code 工具覆盖
    claude_tool_effort: str | None = None,
    claude_tool_allowed: list[str] | None = None,
    # deep_agent 参数
    permissions: list[Any] | None = None,
    backend: Any | None = None,
    subagents: Sequence[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    middleware: Sequence[Any] = (),
    **kwargs: Any,
) -> CompiledStateGraph:
    """创建一个以 Hermes Agent 为 LLM 后端的 LangGraph Agent。

    将 Hermes Agent 的通用 agent 能力（工具调用、记忆、学习）与
    Claude Code CLI 的代码执行能力结合。

    Args:
        model: LLM 后端。
            支持字符串（通过 ChatOpenAI 调用）、
            ChatHermesAgent 实例、或任何 BaseChatModel 实例。
            默认使用 ChatHermesAgent 作为后端。
            （可选）

        tools: 额外的 LangChain 工具。
            （可选）

        mode: 预设模式，决定注册哪些 Claude Code 工具。
            - ``"code_analysis"``（默认）: 只读分析
            - ``"code_refactor"``: 文件读写 + git
            - ``"full_access"``: 无限制访问
            - ``"none"``: 不注册 Claude Code 工具
            （默认 ``"code_analysis"``）

        system_prompt: 自定义系统提示。
            （可选）

        working_dir: Claude Code 工具的工作目录。
            （默认 ``"."``）

        base_url: Hermes Agent API 端点地址。
            （默认 ``"http://localhost:30000/v1"``）

        api_key: Hermes Agent API 密钥。
            （可选）

        hermes_model: Hermes Agent 使用的模型名称。
            （可选，默认使用 Hermes 配置的默认模型）

        provider: Hermes Agent 提供商类型。
            （可选，默认自动检测）

        max_iterations: Hermes Agent 最大工具调用轮次。
            （默认 90）

        enabled_toolsets: Hermes Agent 启用的工具集。
            （可选）

        disabled_toolsets: Hermes Agent 禁用的工具集。
            （可选）

        reasoning_config: Hermes Agent 推理配置。
            （可选）

        claude_tool_effort: 覆盖 Claude Code 工具的 effort 级别。
            （可选）

        claude_tool_allowed: 覆盖 Claude Code 工具的 allowed_tools。
            （可选）

        permissions: deep_agent 文件权限规则。
            （可选）

        backend: deep_agent 后端。
            （可选）

        subagents: 额外的子 agent 列表。
            （可选）

        skills: 技能文件路径列表。
            （可选）

        memory: 记忆文件路径列表。
            （可选）

        interrupt_on: 人工审批配置。
            （可选）

        middleware: 额外的中间件。
            （可选）

        **kwargs: 其它参数透传给 ``deepagents.create_deep_agent``。

    Returns:
        ``CompiledStateGraph``（LangGraph 图）。

    Raises:
        ImportError: deepagents 或 hermes-agent 未安装。
        ValueError: mode 不在已知预设列表中。

    用法示例:

        **模式 A：Hermes + Claude Code 工具** ::

            from hermes_agent import create_hermes_agent

            agent = create_hermes_agent(
                base_url="http://localhost:30000/v1",
                hermes_model="claude-opus-4-20250514",
                mode="code_analysis",
            )
            result = agent.invoke({
                "messages": [{"role": "user", "content": "分析这个项目的架构"}]
            })

        **模式 B：纯 Hermes Agent** ::

            agent = create_hermes_agent(
                base_url="http://localhost:30000/v1",
                hermes_model="claude-opus-4-20250514",
                mode="none",
            )

        **模式 C：已配置的 ChatHermesAgent 实例** ::

            from chat_hermes_agent import ChatHermesAgent
            from hermes_agent import create_hermes_agent

            llm = ChatHermesAgent(
                base_url="http://localhost:30000/v1",
                model="claude-opus-4-20250514",
            )
            agent = create_hermes_agent(model=llm, mode="full_access")
    """
    # ── 校验 ──
    if mode not in _PRESETS:
        raise ValueError(
            f"未知模式: {mode!r}，可选值为: {sorted(_PRESETS.keys())}"
        )

    # ── 模型默认值 ──
    if model is None:
        # 默认使用 ChatHermesAgent 作为后端
        from chat_hermes_agent import ChatHermesAgent

        model = ChatHermesAgent(
            base_url=base_url,
            api_key=api_key,
            model=hermes_model,
            provider=provider,
            max_iterations=max_iterations,
            working_dir=working_dir,
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            reasoning_config=reasoning_config,
        )
    elif isinstance(model, str):
        # 字符串模型名 — 通过 ChatOpenAI 调用
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "使用字符串模型名需要安装 langchain-openai: pip install langchain-openai"
            )
        model = ChatOpenAI(model=model)

    # ── 构建工具列表 ──
    from langchain_core.tools import BaseTool

    all_tools: list[BaseTool] = list(tools) if tools else []

    if mode != "none":
        claude_tools = _make_preset_tool_list(
            mode=mode,
            effort_override=claude_tool_effort,
            allowed_override=claude_tool_allowed,
            working_dir=working_dir,
        )
        all_tools.extend(claude_tools)
        logger.debug("已注册 Claude Code 工具（mode=%s）: %s", mode,
                      [t.name for t in claude_tools])

    # ── 构建系统提示 ──
    prompt = _build_system_prompt(mode, system_prompt)

    # ── 调用 deepagents ──
    try:
        from deepagents import create_deep_agent
    except ImportError:
        raise ImportError(
            "需要安装 deepagents 包: pip install deepagents"
        )

    deep_kwargs: dict[str, Any] = {
        "model": model,
        "tools": all_tools if all_tools else None,
        "system_prompt": prompt,
        "middleware": list(middleware) if middleware else [],
    }

    if permissions is not None:
        deep_kwargs["permissions"] = permissions
    if backend is not None:
        deep_kwargs["backend"] = backend
    if subagents is not None:
        deep_kwargs["subagents"] = subagents
    if skills is not None:
        deep_kwargs["skills"] = skills
    if memory is not None:
        deep_kwargs["memory"] = memory
    if interrupt_on is not None:
        deep_kwargs["interrupt_on"] = interrupt_on

    deep_kwargs.update(kwargs)

    agent = create_deep_agent(**deep_kwargs)

    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", "")
        or type(model).__name__
    )
    logger.info("Hermes Agent 创建成功: model=%s, mode=%s, tools=%d",
                model_name, mode, len(all_tools))

    return agent


# ═══════════════════════════════════════════════════════════════
# 快捷函数
# ═══════════════════════════════════════════════════════════════

def create_hermes_code_analysis_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建 Hermes 代码分析 Agent。

    等同于 ``create_hermes_agent(mode="code_analysis", ...)``。
    """
    return create_hermes_agent(model=model, mode="code_analysis", **kwargs)


def create_hermes_code_refactor_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建 Hermes 代码重构 Agent。

    等同于 ``create_hermes_agent(mode="code_refactor", ...)``。
    """
    return create_hermes_agent(model=model, mode="code_refactor", **kwargs)


def create_hermes_full_access_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建 Hermes 完全访问 Agent。

    等同于 ``create_hermes_agent(mode="full_access", ...)``。
    """
    return create_hermes_agent(model=model, mode="full_access", **kwargs)


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "create_hermes_agent",
    "create_hermes_code_analysis_agent",
    "create_hermes_code_refactor_agent",
    "create_hermes_full_access_agent",
    "_PRESETS",
]
