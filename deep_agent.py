"""
Deep Agent — 将 deep_agent 编排框架与 Claude Code CLI 执行能力结合

deep_agent (LangChain Deep Agent) 负责编排 — 任务规划、todo、子 agent 调度，
Claude Code 负责执行 — 文件操作、代码分析、shell 命令、git 操作。

用法:
    from deep_agent import create_claude_deep_agent

    # 模式 A：标准 LLM 编排 + Claude Code 工具执行（推荐）
    agent = create_claude_deep_agent(model="deepseek-v4-pro", mode="code_analysis")
    result = agent.invoke({"messages": [{"role": "user", "content": "分析项目架构"}]})

    # 模式 B：ChatClaudeCode 本身就是 LLM 后端
    from chat_claude_code import ChatClaudeCode
    agent = create_claude_deep_agent(
        model=ChatClaudeCode(working_dir=".", effort="high"),
        mode="none",
    )

预设模式:
    - code_analysis: 只读分析（默认）
    - code_refactor: 读写代码
    - full_access:  无限制访问
    - none:         纯 deep_agent，不注册 Claude Code 工具

依赖:
    deepagents >= 0.5.0
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
        "description": "不注册 Claude Code 工具 — 纯 deep_agent",
        "register": [],
    },
}


def _make_preset_tool_list(
    mode: str,
    effort_override: str | None = None,
    allowed_override: list[str] | None = None,
    working_dir: str = ".",
) -> list[BaseTool]:
    """根据预设模式构建工具列表"""
    from claude_code_tool import claude_code, claude_code_isolated, claude_code_structured

    preset = _PRESETS[mode]
    tools: list[BaseTool] = []

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


def _copy_tool(t: BaseTool) -> BaseTool:
    """浅拷贝工具（通过重新构造），避免修改原始模板。"""
    return t.model_copy(deep=False) if hasattr(t, "model_copy") else t


def _apply_tool_defaults(tool: BaseTool, defaults: dict[str, Any]) -> BaseTool:
    """为工具参数设置默认值，使其在 Agent 调用时无需手动传递。

    不直接修改原工具，而是创建一个新工具，其包装函数是普通 Python 函数
    （非 functools.partial），确保 LangGraph 的 get_type_hints() 能正常工作。
    """
    from langchain_core.tools import StructuredTool

    original = tool.func

    # 用闭包创建真正的函数（不是 functools.partial），
    # get_type_hints() 对普通函数有效，对 partial 无效
    def _wrapper(**kwargs: Any) -> str:
        """带默认值的工具调用包装。"""
        merged: dict[str, Any] = dict(defaults)
        merged.update(kwargs)
        return original(**merged)

    _wrapper.__name__ = original.__name__
    _wrapper.__qualname__ = original.__qualname__
    _wrapper.__doc__ = tool.description
    _wrapper.__annotations__ = {"return": str}

    # 构建新的 args_schema：已设置默认值的字段标记为可选
    new_schema = None
    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            from pydantic import create_model

            schema = tool.args_schema.model_json_schema()
            fields: dict[str, Any] = {}
            for name_, info in schema.get("properties", {}).items():
                is_required = name_ in schema.get("required", [])
                if name_ in defaults:
                    # 已设默认值 → 可选
                    fields[name_] = (
                        str,
                        defaults[name_],
                    )
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
    """根据模式生成系统提示"""
    if custom_prompt:
        return custom_prompt

    if mode == "none":
        return None  # 使用 deep_agent 默认提示

    preset: dict[str, Any] = _PRESETS[mode]
    description: str = preset["description"]
    registered: list[str] = preset.get("register", [])

    tool_hints = _tool_hints_for_mode(mode)

    return f"""你是一个代码 Agent，由 deep_agent 框架编排，底层通过 Claude Code CLI 执行代码级任务。

**当前模式**：{mode}（{description}）

**可用工具说明**：

- deep_agent 内置工具（`write_todos`、`task`、文件操作、`execute`）用于规划和编排。
{chr(10).join(f'- 已注册 `{t}` 工具：{tool_hints.get(t, "委托代码任务给 Claude Code CLI")}' for t in registered)}

**工作原则**：

1. **先规划再执行**：用 `write_todos` 列出子任务，然后逐步完成。
2. **代码操作委托 Claude Code**：涉及多文件读取、代码分析、重构、git 操作时，优先用 Claude Code 工具而非内置文件工具。Claude Code 工具拥有完整的项目上下文和原生 shell 能力。
3. **简单操作直接用内置工具**：读单个文件、搜索单个符号等轻量操作，用 deep_agent 内置的 `read_file`、`grep`、`glob` 更快。
4. **复杂子任务用 `task`**：将独立的复杂子问题委托给子 agent 并行处理。
5. **结构化提取用 `claude_code_structured`**：需要从代码中提取结构化信息（函数签名、类定义、依赖关系等）时使用。
6. **每次完成后向用户报告进展**，说明已完成的工作和下一步计划。
"""


def _tool_hints_for_mode(mode: str) -> dict[str, str]:
    """返回当前模式下各工具的使用提示"""
    hints: dict[str, str] = {}
    registered = _PRESETS[mode]["register"]

    if "claude_code" in registered:
        hints["claude_code"] = (
            "委托复杂的代码任务（多文件读写、shell 命令、git 操作、代码分析）。"
            "支持 session_id 保持多步上下文。"
        )
    if "claude_code_structured" in registered:
        hints["claude_code_structured"] = (
            "委托任务并要求结构化 JSON 输出。适合提取函数签名、类定义、依赖关系等结构化数据。"
        )
    if "claude_code_isolated" in registered:
        hints["claude_code_isolated"] = (
            "在隔离的 git worktree 中执行任务。适合并行处理互不干扰的子任务。"
        )

    return hints


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def create_claude_deep_agent(
    model: str | BaseChatModel | None = None,
    *,
    tools: Sequence[BaseTool] | None = None,
    mode: str = "code_analysis",
    system_prompt: str | None = None,
    working_dir: str = ".",
    claude_tool_effort: str | None = None,
    claude_tool_allowed: list[str] | None = None,
    permissions: list[Any] | None = None,
    backend: Any | None = None,
    subagents: Sequence[Any] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    interrupt_on: dict[str, Any] | None = None,
    middleware: Sequence[Any] = (),
    **kwargs: Any,
) -> CompiledStateGraph:
    """创建一个集成了 Claude Code CLI 能力的 deep_agent。

    将 deep_agent（LangChain Deep Agent）的编排能力与 Claude Code CLI
    的执行能力结合：deep_agent 负责任务规划、拆解、子 agent 调度；
    Claude Code 负责文件操作、代码分析、shell 命令等执行类任务。

    Args:
        model: LLM 后端。
            支持模型名字符串（如 ``"deepseek-v4-pro"``），
            或任何 ``BaseChatModel`` 实例（包括 ``ChatClaudeCode``）。
            默认使用 ``"deepseek-v4-pro"``。

        tools: 额外的 LangChain 工具，与 Claude Code 工具合并后注册。
            （可选）

        mode: 预设模式，决定注册哪些 Claude Code 工具及其配置。
            - ``"code_analysis"``（默认）: 只读分析和结构化提取
            - ``"code_refactor"``: 文件读写 + git 操作
            - ``"full_access"``: 无限制访问 + 隔离执行
            - ``"none"``: 不注册任何 Claude Code 工具
            （默认 ``"code_analysis"``）

        system_prompt: 自定义系统提示。不传则根据 mode 自动生成。
            （可选）

        working_dir: Claude Code 工具的工作目录。所有 Claude Code
            工具调用都将在此目录下执行。默认 ``"."``（当前目录）。
            （可选）

        claude_tool_effort: 覆盖 Claude Code 工具的 effort 级别。
            可以是 ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``。
            （可选，默认根据 mode 自动设置）

        claude_tool_allowed: 覆盖 ``claude_code`` 的 ``allowed_tools`` 列表。
            可以是工具名列表（如 ``["Read", "Write"]``），
            或 ``[]`` 表示不限制。仅在 mode 不为 ``"none"`` 时生效。
            （可选，默认根据 mode 自动设置）

        permissions: deep_agent 的文件系统权限规则列表。
            （可选，透传给 ``create_deep_agent``）

        backend: deep_agent 后端（文件存储和执行沙箱）。
            （可选，透传给 ``create_deep_agent``）

        subagents: 额外的子 agent 列表。
            （可选，透传给 ``create_deep_agent``）

        skills: 技能文件路径列表。
            （可选，透传给 ``create_deep_agent``）

        memory: 记忆文件路径列表。
            （可选，透传给 ``create_deep_agent``）

        interrupt_on: 人工审批配置。
            （可选，透传给 ``create_deep_agent``）

        middleware: 额外的中间件。
            （可选，透传给 ``create_deep_agent``）

        **kwargs: 其它参数透传给 ``deepagents.create_deep_agent``。

    Returns:
        ``CompiledStateGraph``（LangGraph 图），支持 ``invoke``、``stream``、
        ``ainvoke`` 等标准接口。

    Raises:
        ImportError: deepagents 未安装或版本过低。
        ValueError: mode 不在已知预设列表中。

    用法示例:

        **模式 A：标准 LLM + Claude Code 工具** ::

            from deep_agent import create_claude_deep_agent

            agent = create_claude_deep_agent(
                model="deepseek-v4-pro",
                mode="code_analysis",
            )
            result = agent.invoke({
                "messages": [{"role": "user", "content": "分析这个项目的架构"}]
            })

        **模式 B：ChatClaudeCode 作为 LLM** ::

            from deep_agent import create_claude_deep_agent
            from chat_claude_code import ChatClaudeCode

            agent = create_claude_deep_agent(
                model=ChatClaudeCode(working_dir=".", effort="high"),
                mode="none",
            )
            result = agent.invoke({
                "messages": [{"role": "user", "content": "重构 src/ 下的代码"}]
            })

        **自定义配置** ::

            agent = create_claude_deep_agent(
                model="deepseek-v4-pro",
                mode="code_refactor",
                claude_tool_effort="high",
                claude_tool_allowed=["Read", "Write", "Edit", "Bash(git *)"],
                subagents=[custom_subagent],
            )
    """
    # ── 校验 ──
    if mode not in _PRESETS:
        raise ValueError(
            f"未知模式: {mode!r}，可选值为: {sorted(_PRESETS.keys())}"
        )

    # ── 模型默认值 ──
    if model is None:
        model = "deepseek-v4-pro"

    # 字符串模型名通过 ChatOpenAI 调用（兼容 OpenAI-compatible API），
    # 避免 LangChain init_chat_model() 根据名称自动匹配到
    # ChatDeepSeek / langchain-deepseek 等非预期 provider。
    if isinstance(model, str):
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=model)

    # ── 构建工具列表 ──
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

    logger.info("Deep Agent 创建成功: model=%s, mode=%s, tools=%d",
                getattr(model, "model_name", type(model).__name__),
                mode,
                len(all_tools))

    return agent


# ═══════════════════════════════════════════════════════════════
# 快捷函数
# ═══════════════════════════════════════════════════════════════

def create_code_analysis_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建只读代码分析 Agent。

    等同于 ``create_claude_deep_agent(mode="code_analysis", ...)``。
    """
    return create_claude_deep_agent(model=model, mode="code_analysis", **kwargs)


def create_code_refactor_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建代码重构 Agent。

    等同于 ``create_claude_deep_agent(mode="code_refactor", ...)``。
    """
    return create_claude_deep_agent(model=model, mode="code_refactor", **kwargs)


def create_full_access_agent(
    model: str | BaseChatModel | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """快捷函数：创建完全访问 Agent。

    等同于 ``create_claude_deep_agent(mode="full_access", ...)``。
    """
    return create_claude_deep_agent(model=model, mode="full_access", **kwargs)


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "create_claude_deep_agent",
    "create_code_analysis_agent",
    "create_code_refactor_agent",
    "create_full_access_agent",
    "_PRESETS",
]
