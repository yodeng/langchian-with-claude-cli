"""
ChatHermesAgent 使用示例

演示 ChatHermesAgent 的 8 种使用模式：

1. 基础单轮调用
2. 流式输出
3. 多轮对话（自动 session 管理）
4. bind_tools() 工具绑定
5. with_structured_output() 结构化输出
6. 在 LangGraph 中使用
7. 自定义 Hermes 配置
8. Tool 形式（@tool 委托）

启动方式:
    python examples/example_hermes_chat_model.py

前置条件:
    - Hermes Agent 已安装并配置（pip install hermes-agent，或设置 HERMES_HOME）
    - Hermes Agent API 已启动（默认 http://localhost:30000/v1）
"""

from __future__ import annotations

import os
import sys

# 确保项目根在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chat_hermes_agent import ChatHermesAgent


def demo_1_basic():
    """示例 1：基础单轮调用"""
    print("=" * 60)
    print("示例 1：基础单轮调用")
    print("=" * 60)

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    response = llm.invoke("用一句话解释什么是 Python 装饰器")
    print(f"回复: {response.content}")
    print()


def demo_2_streaming():
    """示例 2：流式输出"""
    print("=" * 60)
    print("示例 2：流式输出")
    print("=" * 60)

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    print("流式输出: ", end="", flush=True)
    for chunk in llm.stream("写一个 Python 快速排序函数"):
        print(chunk.content, end="", flush=True)
    print("\n")


def demo_3_multi_turn():
    """示例 3：多轮对话（自动 session 管理）"""
    print("=" * 60)
    print("示例 3：多轮对话")
    print("=" * 60)

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    # 第一轮
    response = llm.invoke("我的名字叫小明，是一名 Python 开发者")
    print(f"第1轮: {response.content[:100]}...")

    # 第二轮 — 自动在同一 session 内延续
    response = llm.invoke("我叫什么名字？我的职业是什么？")
    print(f"第2轮: {response.content[:200]}...")

    # 重置会话
    llm.reset_session()
    print("(会话已重置)")

    # 第三轮 — 新 session，不记得之前的对话
    response = llm.invoke("我叫什么名字？")
    print(f"第3轮 (新会话): {response.content[:200]}...")
    print()


def demo_4_bind_tools():
    """示例 4：bind_tools() 工具绑定"""
    print("=" * 60)
    print("示例 4：bind_tools() 工具绑定")
    print("=" * 60)

    from langchain_core.tools import tool

    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的天气信息"""
        weather_data = {
            "北京": "晴天，25°C，湿度 45%",
            "上海": "多云，28°C，湿度 60%",
            "深圳": "阵雨，30°C，湿度 75%",
        }
        return weather_data.get(city, f"未找到{city}的天气数据")

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    # 绑定工具
    llm_with_tools = llm.bind_tools([get_weather])

    response = llm_with_tools.invoke("北京的天气怎么样？")
    print(f"回复: {response.content[:300]}...")
    print()


def demo_5_structured_output():
    """示例 5：with_structured_output() 结构化输出"""
    print("=" * 60)
    print("示例 5：with_structured_output()")
    print("=" * 60)

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "函数名称"},
            "parameters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                    },
                },
            },
            "return_type": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["name", "parameters", "return_type", "description"],
    }

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
    )

    llm_structured = llm.with_structured_output(schema)

    response = llm_structured.invoke(
        "分析这个 Python 函数签名并提取结构化信息:\n"
        "def calculate_total(items: list[dict], tax_rate: float = 0.13) -> float:"
    )
    print(f"结构化输出: {response.content[:500]}...")
    print()


def demo_6_langgraph():
    """示例 6：在 LangGraph 中使用"""
    print("=" * 60)
    print("示例 6：在 LangGraph 中使用")
    print("=" * 60)

    try:
        from langgraph.graph import StateGraph, END
        from typing import Annotated, TypedDict
        from langchain_core.messages import BaseMessage
        import operator

        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], operator.add]

        llm = ChatHermesAgent(
            base_url="http://localhost:30000/v1",
            model="claude-opus-4-20250514",
            working_dir=".",
        )

        def agent_node(state: AgentState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.set_entry_point("agent")
        graph.add_edge("agent", END)
        compiled = graph.compile()

        result = compiled.invoke({
            "messages": [{"role": "user", "content": "用一句话解释什么是 Git"}]
        })
        print(f"LangGraph 回复: {result['messages'][-1].content[:200]}...")
    except ImportError as e:
        print(f"跳过（缺少依赖）: {e}")
    print()


def demo_7_custom_config():
    """示例 7：自定义 Hermes 配置"""
    print("=" * 60)
    print("示例 7：自定义 Hermes 配置")
    print("=" * 60)

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        api_key=os.environ.get("HERMES_API_KEY"),
        model="claude-opus-4-20250514",
        provider="anthropic",
        max_iterations=30,
        enabled_toolsets=["terminal", "file"],
        working_dir=".",
        system_prompt="你是一个 Python 代码审查专家。回答要简洁专业。",
        reasoning_config={"effort": "medium"},
    )

    response = llm.invoke("如何优化 Python 列表操作？")
    print(f"回复: {response.content[:300]}...")
    print()


def demo_8_tool_form():
    """示例 8：Tool 形式 — 将 Hermes 作为 LangChain @tool 使用"""
    print("=" * 60)
    print("示例 8：Tool 形式（@tool 委托）")
    print("=" * 60)

    from chat_hermes_agent import (
        hermes_agent,
        hermes_agent_structured,
        hermes_agent_session,
        close_hermes_session,
        list_hermes_sessions,
    )

    # ============================================================
    # 用法 1：基础委托 — 将单个任务委托给 Hermes Agent
    # ============================================================
    print("\n-- 用法 1：基础委托 --")
    result = hermes_agent.invoke({
        "task": "用一句话解释什么是 Docker",
        "model": "claude-opus-4-20250514",
        "base_url": "http://localhost:30000/v1",
    })
    print(f"结果: {result[:200]}...")

    # ============================================================
    # 用法 2：结构化提取 — 要求 JSON 格式输出
    # ============================================================
    print("\n-- 用法 2：结构化提取 --")
    schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string"},
            "frameworks": {"type": "array", "items": {"type": "string"}},
            "strengths": {"type": "array", "items": {"type": "string"}},
        },
    }
    result = hermes_agent_structured.invoke({
        "task": "描述 Python 语言的特点",
        "output_schema": schema,
        "base_url": "http://localhost:30000/v1",
    })
    print(f"结构化结果: {result[:300]}...")

    # ============================================================
    # 用法 3：多步骤会话 — 跨多次调用保持上下文
    # ============================================================
    print("\n-- 用法 3：多步骤会话 --")
    SESSION = "demo-session-001"

    # 第一步
    r1 = hermes_agent_session.invoke({
        "task": "分析 Python 中列表推导式的性能优势",
        "session_id": SESSION,
        "base_url": "http://localhost:30000/v1",
    })

    # 第二步 — 继续同一会话，有上下文记忆
    r2 = hermes_agent_session.invoke({
        "task": "基于你刚才的分析，写一个对比示例",
        "context": r1[:500],
        "session_id": SESSION,
        "base_url": "http://localhost:30000/v1",
    })
    print(f"第1步: {r1[:100]}...")
    print(f"第2步: {r2[:100]}...")

    # 清理
    close_hermes_session(SESSION)
    print(f"活跃会话: {list_hermes_sessions()}")

    # ============================================================
    # 用法 4：在 LangGraph Agent 中使用
    # ============================================================
    print("\n-- 用法 4：在 LangGraph Agent 中使用 --")
    try:
        from langgraph.prebuilt import ToolNode
        from langgraph.graph import StateGraph, END
        from typing import Annotated, TypedDict
        from langchain_core.messages import BaseMessage
        from langchain_openai import ChatOpenAI
        import operator

        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], operator.add]

        tools = [hermes_agent, hermes_agent_structured]
        llm = ChatOpenAI(model="deepseek-v4-pro").bind_tools(tools)

        def agent_node(state):
            return {"messages": [llm.invoke(state["messages"])]}

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", lambda s: (
            "tools" if s["messages"][-1].tool_calls else END
        ))
        graph.add_edge("tools", "agent")
        compiled = graph.compile()
        print("LangGraph Agent (含 hermes_agent 工具) 已构建完成")
    except ImportError:
        print("跳过 (缺少 langgraph / langchain_openai)")
    print()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ChatHermesAgent 使用示例")
    print("前置条件: Hermes Agent API 运行在 http://localhost:30000/v1")
    print()

    examples = [
        ("1", "基础单轮调用", demo_1_basic),
        ("2", "流式输出", demo_2_streaming),
        ("3", "多轮对话", demo_3_multi_turn),
        ("4", "bind_tools() 工具绑定", demo_4_bind_tools),
        ("5", "with_structured_output()", demo_5_structured_output),
        ("6", "在 LangGraph 中使用", demo_6_langgraph),
        ("7", "自定义 Hermes 配置", demo_7_custom_config),
        ("8", "Tool 形式（@tool 委托）", demo_8_tool_form),
    ]

    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print()

    choice = input("选择要运行的示例 (1-8, 或 'all' 全部运行): ").strip()

    if choice.lower() == "all":
        for _, _, func in examples:
            try:
                func()
            except Exception as e:
                print(f"[跳过] {func.__name__}: {e}")
    elif choice in [e[0] for e in examples]:
        try:
            examples[int(choice) - 1][2]()
        except Exception as e:
            print(f"[错误] {e}")
    else:
        print("无效选择")
