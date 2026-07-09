"""
ChatClaudeCode 示例 — 展示所有使用模式

运行：python examples/example_chat_model.py [1|2|3|4|5|6]
  1: 基础调用 (invoke)
  2: 多轮对话
  3: 流式输出
  4: 工具绑定
  5: LangGraph 集成
  6: 异步调用
  7: 全部功能综合测试
"""

import asyncio
import json
import os
import sys

# 确保能导入父目录的 chat_claude_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from chat_claude_code import ChatClaudeCode


# ═══════════════════════════════════════════════════════════════
# 示例 1：基础调用
# ═══════════════════════════════════════════════════════════════

def example_basic():
    """最简单的调用：invoke + 单条消息"""
    print("=" * 60)
    print("示例 1: 基础调用 (invoke)")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)

    # 方式 1：直接传字符串
    result = llm.invoke("用一句话介绍 Python 的优点")
    print(f"\n[方式1 - 字符串] {result.content}")

    # 方式 2：传消息列表（支持 SystemMessage）
    result = llm.invoke([
        SystemMessage(content="你是一个简洁的 Python 专家，只用一个词回答问题"),
        HumanMessage(content="Python 的核心理念是什么？"),
    ])
    print(f"\n[方式2 - 消息列表] {result.content}")

    # 方式 3：多消息混合
    result = llm.invoke([
        SystemMessage(content="你是代码顾问"),
        HumanMessage(content="我有一个问题"),
        AIMessage(content="请说，我来帮你"),
        HumanMessage(content="dict 和 list 的区别是什么？"),
    ])
    print(f"\n[方式3 - 多消息] {result.content[:200]}...")


# ═══════════════════════════════════════════════════════════════
# 示例 2：多轮对话
# ═══════════════════════════════════════════════════════════════

def example_multiturn():
    """多轮对话：自动保持上下文"""
    print("\n" + "=" * 60)
    print("示例 2: 多轮对话")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    print(f"Session ID: {llm._session_id}\n")

    # 第一轮
    r1 = llm.invoke("我正在写一个 Python 函数来解析 JSON，函数名是 parse_config")
    print(f"[Turn 1] {r1.content[:150]}...\n")

    # 第二轮 — 自动在同一 session 内延续，记住上下文
    r2 = llm.invoke("我的函数名是什么？用一个词回答")
    print(f"[Turn 2] {r2.content}\n")

    # 第三轮
    r3 = llm.invoke("给它加一个 validate 参数，只回答 '已添加'")
    print(f"[Turn 3] {r3.content}\n")

    # 重置会话 — 丢失上下文
    llm.reset_session()
    print(f"Session 已重置，新 ID: {llm._session_id}")

    r4 = llm.invoke("我的函数名是什么？")
    print(f"[Turn 4 - 新会话] {r4.content[:150]}...")

    print(f"\n总共 {llm._session_turn} 轮对话")


# ═══════════════════════════════════════════════════════════════
# 示例 3：流式输出
# ═══════════════════════════════════════════════════════════════

def example_streaming():
    """流式输出：边生成边输出"""
    print("\n" + "=" * 60)
    print("示例 3: 流式输出")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)

    print("\n[逐字输出] ", end="")
    chunk_count = 0
    for chunk in llm.stream("从 1 数到 5，用逗号分隔"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            chunk_count += 1
    print(f"\n(共 {chunk_count} 个 chunk)\n")


# ═══════════════════════════════════════════════════════════════
# 示例 4：工具绑定
# ═══════════════════════════════════════════════════════════════

@tool
def read_file(path: str) -> str:
    """读取指定路径的文件内容"""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {e}"


@tool
def count_lines(text: str) -> int:
    """统计文本行数"""
    return len(text.strip().split("\n"))


def example_tool_binding():
    """工具绑定：将 LangChain Tools 绑定到 ChatClaudeCode"""
    print("\n" + "=" * 60)
    print("示例 4: 工具绑定")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)

    # 绑定工具
    llm_with_tools = llm.bind_tools([read_file, count_lines])
    print(f"已绑定 {len(llm_with_tools._bound_tools)} 个工具")
    print(f"System prompt 包含工具描述...\n")

    result = llm_with_tools.invoke(
        "读取 chat_claude_code.py 文件的前 5 行，然后统计其中有多少行是 import 语句。"
        "只需要用工具来完成，返回统计结果。"
    )
    print(f"[工具绑定结果] {result.content[:300]}...")


# ═══════════════════════════════════════════════════════════════
# 示例 5：LangGraph Agent 集成
# ═══════════════════════════════════════════════════════════════

def example_langgraph():
    """在 LangGraph 中使用 ChatClaudeCode 作为 llm 后端"""
    print("\n" + "=" * 60)
    print("示例 5: LangGraph Agent 集成")
    print("=" * 60)

    from typing import Annotated, TypedDict

    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    # ChatClaudeCode 作为 Agent 的 llm
    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=120)

    def agent_node(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.set_entry_point("agent")
    builder.add_edge("agent", END)
    graph = builder.compile()

    result = graph.invoke({
        "messages": [HumanMessage(content="列出当前目录下的文件，统计文件数量。最终只输出数量。")],
    })

    print(f"\n[Agent 输出] {result['messages'][-1].content}")


# ═══════════════════════════════════════════════════════════════
# 示例 6：异步调用
# ═══════════════════════════════════════════════════════════════

async def example_async():
    """异步调用"""
    print("\n" + "=" * 60)
    print("示例 6: 异步调用")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)

    # ainvoke
    result = await llm.ainvoke("1+1=? 只输出数字")
    print(f"\n[ainvoke] {result.content}")

    # astream
    print("[astream] ", end="")
    async for chunk in llm.astream("从 A 到 C，大写字母"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


# ═══════════════════════════════════════════════════════════════
# 示例 7：综合测试
# ═══════════════════════════════════════════════════════════════

def example_comprehensive():
    """运行所有可同步执行的示例"""
    example_basic()
    example_multiturn()
    example_streaming()
    example_tool_binding()
    example_langgraph()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

EXAMPLES = {
    "1": ("基础调用", example_basic),
    "2": ("多轮对话", example_multiturn),
    "3": ("流式输出", example_streaming),
    "4": ("工具绑定", example_tool_binding),
    "5": ("LangGraph 集成", example_langgraph),
    "6": ("异步调用", lambda: asyncio.run(example_async())),
    "7": ("综合测试", example_comprehensive),
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in EXAMPLES:
        name, fn = EXAMPLES[sys.argv[1]]
        print(f"运行示例 {sys.argv[1]}: {name}")
        fn()
    else:
        print("用法: python example_chat_model.py [1-7]\n")
        for k, (name, _) in EXAMPLES.items():
            print(f"  {k}: {name}")
        print(f"\n默认运行示例 1（基础调用）...\n")
        example_basic()
