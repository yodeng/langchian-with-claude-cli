"""
LangGraph Agent 委托 Claude Code 示例

演示三种委托模式：
  1. 单步委托 — 将某个步骤完全交给 Claude Code
  2. 带会话的委托 — 多步骤共享上下文
  3. 结构化委托 — 从代码库提取结构化数据

运行：python examples/example_tool.py
"""

import json
import os
import sys
from typing import Annotated, TypedDict

# 确保能导入父目录的 claude_code_tool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from claude_code_tool import (
    delegate_to_claude_code,
    delegate_to_claude_code_structured,
    delegate_to_claude_code_isolated,
    close_session,
)

# ═══════════════════════════════════════════
# Agent State
# ═══════════════════════════════════════════

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_step: str
    context: dict


# ═══════════════════════════════════════════
# 示例 1：基础委托 Agent
# ═══════════════════════════════════════════

def example_basic_delegation():
    """
    场景：用户要求分析项目结构并生成报告。
    LangChain Agent 负责理解用户意图、编排流程，
    实际的文件分析和代码生成委托给 Claude Code。
    """
    print("=" * 60)
    print("示例 1: 基础委托 — 代码分析 + 报告生成")
    print("=" * 60)

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
    )

    tools = [delegate_to_claude_code]
    llm_with_tools = llm.bind_tools(tools)

    # 构建 Graph
    builder = StateGraph(AgentState)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile()

    # 运行
    result = graph.invoke({
        "messages": [{
            "role": "user",
            "content": """
请帮我完成以下任务（分两步委托给 Claude Code）：

1. 首先，分析 /home/dengyong 目录下的 Python 文件结构和主要功能
2. 然后，基于分析结果生成一份项目架构报告，保存为 project_report.md

使用 delegate_to_claude_code 工具执行每个步骤。
""",
        }],
    })

    print("\n[Agent 最终回复]")
    print(result["messages"][-1].content)


# ═══════════════════════════════════════════
# 示例 2：多步会话委托
# ═══════════════════════════════════════════

def example_session_delegation():
    """
    场景：多步骤任务需要共享上下文。
    比如：先探索代码 → 识别问题 → 修复问题 → 验证修复。
    每个步骤在同一 Claude Code session 内执行，保持上下文。
    """
    print("\n" + "=" * 60)
    print("示例 2: 会话委托 — 发现→修复→验证 全流程")
    print("=" * 60)

    SESSION_ID = "demo-session-001"

    # 步骤 1：探索代码库
    print("\n--- 步骤 1: 探索代码库 ---")
    r1 = delegate_to_claude_code.invoke({
        "task": "探索 /home/dengyong 目录，列出所有 Python 文件，"
                "简要说明每个文件的功能。不需要修改任何文件。",
        "session_id": SESSION_ID,
        "allowed_tools": ["Read", "Bash(ls *)", "Bash(find *)"],
        "effort": "low",
    })
    print(r1[:500] + "..." if len(r1) > 500 else r1)

    # 步骤 2：在同一个 session 中修复问题
    print("\n--- 步骤 2: 修复问题（同一会话） ---")
    r2 = delegate_to_claude_code.invoke({
        "task": "基于你刚才对项目的理解，检查 rag_prompts.py 是否有可改进的地方，"
                "如果有，请优化并保存。如果没有，说明原因。",
        "session_id": SESSION_ID,
        "allowed_tools": ["Read", "Write", "Edit"],
        "effort": "medium",
    })
    print(r2[:500] + "..." if len(r2) > 500 else r2)

    # 步骤 3：验证修复
    print("\n--- 步骤 3: 验证修复（同一会话） ---")
    r3 = delegate_to_claude_code.invoke({
        "task": "确认你刚才做的修改是否正确：重新读取 rag_prompts.py，"
                "验证语法和逻辑正确性。",
        "session_id": SESSION_ID,
        "allowed_tools": ["Read", "Bash(python *)"],
        "effort": "low",
    })
    print(r3[:500] + "..." if len(r3) > 500 else r3)

    # 清理会话
    close_session(SESSION_ID)
    print("\n✅ 会话委托完成")


# ═══════════════════════════════════════════
# 示例 3：结构化输出委托
# ═══════════════════════════════════════════

def example_structured_delegation():
    """
    场景：从代码库提取结构化信息。
    比如：扫描所有 Python 文件，提取函数签名、类定义和依赖关系。
    """
    print("\n" + "=" * 60)
    print("示例 3: 结构化委托 — 提取 API 清单")
    print("=" * 60)

    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "文件路径"},
                        "language": {"type": "string"},
                        "functions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "signature": {"type": "string"},
                                    "docstring": {"type": "string"},
                                    "line_count": {"type": "integer"},
                                },
                                "required": ["name", "signature"],
                            },
                        },
                        "classes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "methods": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "imports": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["path", "functions", "imports"],
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "total_files": {"type": "integer"},
                    "total_functions": {"type": "integer"},
                    "total_classes": {"type": "integer"},
                    "key_dependencies": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "required": ["files", "summary"],
    }

    result = delegate_to_claude_code_structured.invoke({
        "task": (
            "扫描 /home/dengyong 目录下所有 .py 文件（不包括 miniconda 和 .cache 子目录）。"
            "对每个文件，提取：函数名+签名、类名+方法、import 列表。"
            "最后输出 summary，汇总文件数、函数数、类数、关键依赖。"
            "只读取分析，不要修改任何文件。"
        ),
        "output_schema": schema,
        "effort": "medium",
    })

    print(result)

    # 尝试解析为 Python 对象使用
    try:
        data = json.loads(result)
        print(f"\n📊 汇总: {data.get('summary', {}).get('total_files', 0)} 个文件, "
              f"{data.get('summary', {}).get('total_functions', 0)} 个函数, "
              f"{data.get('summary', {}).get('total_classes', 0)} 个类")
    except json.JSONDecodeError:
        print("\n⚠️ 解析失败，使用原始输出")


# ═══════════════════════════════════════════
# 示例 4：完整 LangGraph Agent（组合三种委托模式）
# ═══════════════════════════════════════════

SYSTEM_PROMPT = """你是一个技术主管 Agent，负责协调任务执行。

你有以下工具可以调用：
- delegate_to_claude_code: 将单步任务委派给 Claude Code（文件操作、代码分析、shell 执行等）
- delegate_to_claude_code_structured: 委派并获取结构化 JSON 数据
- delegate_to_claude_code_isolated: 在隔离环境中委派任务

工作流程：
1. 分析用户需求，拆解为子任务
2. 对每个子任务，选择合适的委托工具
3. 汇总结果并返回给用户

重要规则：
- 涉及文件读写、git、shell 命令的任务，必须委托给 Claude Code
- 需要结构化输出（如统计数据）时，用 delegate_to_claude_code_structured
- 多步骤相关任务使用相同的 session_id 保持上下文
- 每次调用后，向用户报告进展
"""


def example_full_agent():
    """
    场景：用户要求 '检查项目代码质量并给出改进建议'
    Agent 自动拆解为：
      1. 代码结构分析 → delegate_to_claude_code_structured
      2. Lint/格式化检查 → delegate_to_claude_code
      3. 生成改进报告 → delegate_to_claude_code（同一 session）
    """
    print("\n" + "=" * 60)
    print("示例 4: 完整 Agent — 代码质量审查")
    print("=" * 60)

    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    tools = [
        delegate_to_claude_code,
        delegate_to_claude_code_structured,
        delegate_to_claude_code_isolated,
    ]
    llm_with_tools = llm.bind_tools(tools)

    builder = StateGraph(AgentState)

    def agent_node(state: AgentState):
        messages = state["messages"]
        if not any(m.get("role") == "system" for m in messages if isinstance(m, dict)):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else END,
    )
    builder.add_edge("tools", "agent")

    graph = builder.compile()

    result = graph.invoke({
        "messages": [{
            "role": "user",
            "content": """
检查 /home/dengyong 下 Python 项目的代码质量，给出改进建议：

1. 先用 delegate_to_claude_code_structured 分析代码结构和依赖关系
2. 再用 delegate_to_claude_code 检查代码风格、潜在问题
3. 最后用 delegate_to_claude_code（同一 session）生成一份改进报告保存为 quality_report.md
""",
        }],
    })

    print("\n[Agent 最终回复]")
    print(result["messages"][-1].content)


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import sys

    examples = {
        "1": ("基础委托", example_basic_delegation),
        "2": ("会话委托", example_session_delegation),
        "3": ("结构化委托", example_structured_delegation),
        "4": ("完整 Agent", example_full_agent),
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        _, fn = examples[sys.argv[1]]
        fn()
    else:
        print("用法: python example_tool.py [1|2|3|4]\n")
        for k, (name, _) in examples.items():
            print(f"  {k}: {name}")
        print("\n不加参数则运行示例 2（会话委托）...\n")
        example_session_delegation()
