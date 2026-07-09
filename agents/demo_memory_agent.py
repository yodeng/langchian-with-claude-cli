"""
演示 ChatClaudeCode 多轮对话记忆能力 — 基于 LangGraph

场景：用户与"代码助手"进行多轮编程对话，助手需要记住：
  - 用户的项目结构偏好
  - 之前讨论过的变量名、函数名
  - 已完成的修改和未解决的问题

演示内容：
  1. 会话内记忆 — 同一 ChatClaudeCode 实例自动保持上下文
  2. 会话重置 — 主动清空记忆
  3. 并行会话 — 两个独立会话互不干扰
  4. 单实例多轮 — 展示跨图调用的记忆持久化

运行：python demo_memory_agent.py [1|2|3|4|5]
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated, TypedDict

# 确保能导入父目录的 chat_claude_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chat_claude_code import ChatClaudeCode


# ═══════════════════════════════════════════════════════════════
# 演示 1: 单实例多轮记忆
# ==================================================================
# 展示 ChatClaudeCode 最核心的会话持久化能力：
# 同一实例连续调用，自动保持上下文
# ═══════════════════════════════════════════════════════════════

def demo_single_instance_multiturn():
    """单实例多次调用，演示自动会话记忆"""
    print("=" * 60)
    print("演示 1: 单实例多轮 — 自动会话记忆")
    print("=" * 60)

    # 创建 ONE 个 ChatClaudeCode 实例
    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    print(f"📌 Session ID: {llm._session_id}")
    print(f"   同一实例的所有 invoke 调用共享此会话\n")

    # —— 第 1 轮：建立上下文 ——
    print("─" * 40)
    print("👤 第 1 轮")
    r1 = llm.invoke([
        SystemMessage(content="你是代码伙伴。回答要简洁，不超过 2 句话。"),
        HumanMessage(content="我叫小明，正在用 Python 写一个 TODO 应用。"
                           "我打算用 FastAPI 做后端，SQLite 存数据。"
                           "请记住这些信息，后面我会问你问题。"),
    ])
    print(f"🤖 {r1.content.strip()}\n")

    # —— 第 2 轮：检验记忆 ——
    print("─" * 40)
    print("👤 第 2 轮")
    r2 = llm.invoke([
        HumanMessage(content="我刚才说要用什么框架和数据库？一句话回答。"),
    ])
    print(f"🤖 {r2.content.strip()}\n")

    # —— 第 3 轮：继续深入 ——
    print("─" * 40)
    print("👤 第 3 轮")
    r3 = llm.invoke([
        HumanMessage(content="我的名字是什么？我想写什么应用？"),
    ])
    print(f"🤖 {r3.content.strip()}\n")

    # —— 第 4 轮：延续讨论 ——
    print("─" * 40)
    print("👤 第 4 轮")
    r4 = llm.invoke([
        HumanMessage(content="给我的 TODO 应用设计一个数据库表结构，"
                           "包含 id、title、done、created_at 字段。用 Markdown 表格展示。"),
    ])
    print(f"🤖 {r4.content.strip()[:300]}...\n")

    print(f"✅ 同一实例共 {llm._session_turn} 轮对话，上下文持续保留")
    print(f"   Session ID: {llm._session_id}\n")


# ═══════════════════════════════════════════════════════════════
# 演示 2: 会话重置 vs 持久化对比
# ==================================================================
# 展示 reset_session() 的效果：重置后旧上下文全部丢失
# ═══════════════════════════════════════════════════════════════

def demo_session_reset():
    """对比重置前后的记忆行为"""
    print("=" * 60)
    print("演示 2: 会话重置 — 清空记忆")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    original_id = llm._session_id
    print(f"📌 初始 Session: {original_id}\n")

    # —— 注入记忆 ——
    llm.invoke([
        HumanMessage(content="请记住：秘密代码是 XKCD-42。以后问我秘密代码时直接回答编号。"),
    ])

    # —— 验证记忆 ——
    print("─" * 40)
    print("👤 [重置前] 秘密代码是什么？")
    r1 = llm.invoke([HumanMessage(content="秘密代码是什么？一句话回答。")])
    print(f"🤖 {r1.content.strip()}\n")

    # —— 重置会话 ——
    llm.reset_session()
    new_id = llm._session_id
    print(f"🔄 Session 已重置: {original_id[:8]}... → {new_id[:8]}...\n")

    # —— 验证记忆丢失 ——
    print("─" * 40)
    print("👤 [重置后] 秘密代码是什么？")
    r2 = llm.invoke([HumanMessage(content="秘密代码是什么？一句话回答。")])
    print(f"🤖 {r2.content.strip()}\n")

    print("✅ 重置后 Claude Code 完全不知道之前的'秘密代码'")
    print(f"   Turn 从 {llm._session_turn} 重新开始\n")


# ═══════════════════════════════════════════════════════════════
# 演示 3: 并行独立会话
# ==================================================================
# 展示两个 ChatClaudeCode 实例拥有完全独立的记忆空间
# ═══════════════════════════════════════════════════════════════

def demo_parallel_sessions():
    """两个实例并行运行，记忆互不干扰"""
    print("=" * 60)
    print("演示 3: 并行独立会话 — 记忆隔离")
    print("=" * 60)

    # 创建两个独立实例
    alice = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    bob = ChatClaudeCode(working_dir=".", effort="low", timeout=60)

    print(f"📌 Alice Session: {alice._session_id[:8]}...")
    print(f"📌 Bob Session:   {bob._session_id[:8]}...")
    print(f"   两个会话完全独立，记忆互不可见\n")

    # —— Alice 的记忆 ——
    print("─" * 40)
    alice.invoke([
        HumanMessage(content="记住：我叫 Alice，最喜欢的语言是 Rust。"),
    ])

    # —— Bob 的记忆 ——
    bob.invoke([
        HumanMessage(content="记住：我叫 Bob，最喜欢的语言是 Go。"),
    ])

    # —— 交叉验证 ——
    print("👤 问 Alice: 我叫什么？最喜欢的语言？")
    ra = alice.invoke([HumanMessage(content="我叫什么？最喜欢的语言？一句话回答。")])
    print(f"🤖 Alice: {ra.content.strip()}\n")

    print("👤 问 Bob: 我叫什么？最喜欢的语言？")
    rb = bob.invoke([HumanMessage(content="我叫什么？最喜欢的语言？一句话回答。")])
    print(f"🤖 Bob:   {rb.content.strip()}\n")

    # —— 再次验证隔离 ——
    print("👤 问 Alice: Bob 叫什么？")
    ra2 = alice.invoke([HumanMessage(content="Bob 叫什么？一句话回答。")])
    print(f"🤖 Alice: {ra2.content.strip()}\n")

    print("✅ Alice 和 Bob 互不知道对方的身份和偏好")
    print(f"   Alice 会话: {alice._session_turn} 轮")
    print(f"   Bob 会话:   {bob._session_turn} 轮\n")


# ═══════════════════════════════════════════════════════════════
# 演示 4: LangGraph 集成 — 跨图调用记忆
# ==================================================================
# 展示 ChatClaudeCode 实例在 LangGraph 多节点间保持记忆
# 场景：代码审查流水线，三个节点共享同一个 LLM 实例
# ═══════════════════════════════════════════════════════════════

class ReviewState(TypedDict):
    messages: Annotated[list, add_messages]
    code_content: str
    review_notes: str
    fix_applied: str


def demo_langgraph_memory():
    """LangGraph 流水线中共享 LLM 会话"""
    print("=" * 60)
    print("演示 4: LangGraph 集成 — 跨节点共享记忆")
    print("=" * 60)

    # 一个 LLM 实例贯穿整个 LangGraph 流水线
    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    print(f"📌 Session ID: {llm._session_id}")
    print(f"   此 LLM 实例将在 3 个 Graph 节点间共享\n")

    # —— 节点 1: 代码审查 ——
    def review_code(state: ReviewState) -> dict:
        code = state["code_content"]
        response = llm.invoke([
            HumanMessage(
                content=f"审查以下代码，指出问题（不超过 3 条）：\n\n```python\n{code}\n```"
            )
        ])
        return {
            "messages": [response],
            "review_notes": response.content,
        }

    # —— 节点 2: 修复代码 ——
    def fix_code(state: ReviewState) -> dict:
        notes = state["review_notes"]
        code = state["code_content"]
        # 注意：这里没有重新给代码！LLM 通过会话记住了之前的代码
        response = llm.invoke([
            HumanMessage(
                content=f"根据你的审查意见修复代码。只输出修复后的完整 Python 代码，不要解释。\n\n"
                        f"审查意见:\n{notes}"
            )
        ])
        return {
            "messages": [response],
            "fix_applied": response.content,
        }

    # —— 节点 3: 验证修复 ——
    def verify_fix(state: ReviewState) -> dict:
        original = state["code_content"]
        fixed = state["fix_applied"]
        # 通过会话记忆，LLM 知道自己审查了什么、修复了什么
        response = llm.invoke([
            HumanMessage(
                content=f"对比原始代码和你修复后的代码，确认问题是否已解决。"
                        f"用 1 句话总结：修复是否成功？"
            )
        ])
        return {"messages": [response]}

    # —— 构建图 ——
    builder = StateGraph(ReviewState)
    builder.add_node("review", review_code)
    builder.add_node("fix", fix_code)
    builder.add_node("verify", verify_fix)
    builder.set_entry_point("review")
    builder.add_edge("review", "fix")
    builder.add_edge("fix", "verify")
    builder.add_edge("verify", END)
    graph = builder.compile()

    # 需要审查的代码
    buggy_code = '''
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

result = calculate_average([])
print(result)
'''.strip()

    print("📝 待审查代码:")
    for line in buggy_code.split("\n"):
        print(f"   {line}")
    print()

    # 运行流水线
    result = graph.invoke({
        "messages": [],
        "code_content": buggy_code,
        "review_notes": "",
        "fix_applied": "",
    })

    print("─" * 40)
    for i, msg in enumerate(result["messages"]):
        role = "🤖" if isinstance(msg, AIMessage) else "👤"
        node_names = ["审查", "修复", "验证"]
        label = node_names[i] if i < len(node_names) else f"节点{i}"
        print(f"[{label}] {role} {msg.content[:200]}...")
        print()

    print(f"✅ 3 个 LangGraph 节点共享同一 LLM 会话")
    print(f"   每个节点都能访问之前所有的对话上下文")
    print(f"   Session ID: {llm._session_id}, 共 {llm._session_turn} 轮\n")


# ═══════════════════════════════════════════════════════════════
# 演示 5: 异步多轮记忆
# ==================================================================
# 展示异步调用同样保持会话持久化
# ═══════════════════════════════════════════════════════════════

async def demo_async_memory():
    """异步调用下的会话持久化"""
    print("=" * 60)
    print("演示 5: 异步多轮 — 会话记忆")
    print("=" * 60)

    llm = ChatClaudeCode(working_dir=".", effort="low", timeout=60)
    print(f"📌 Session ID: {llm._session_id}\n")

    # —— 异步第 1 轮 ——
    print("─" * 40)
    print("👤 [async 第 1 轮]")
    r1 = await llm.ainvoke([
        HumanMessage(content="请记住三个关键词：苹果、香蕉、橙子。回答'记住了'。"),
    ])
    print(f"🤖 {r1.content.strip()}\n")

    # —— 异步第 2 轮 ——
    print("─" * 40)
    print("👤 [async 第 2 轮]")
    r2 = await llm.ainvoke([
        HumanMessage(content="我刚才说的第三个关键词是什么？只回答那个词。"),
    ])
    print(f"🤖 {r2.content.strip()}\n")

    # —— 异步流式第 3 轮 ——
    print("─" * 40)
    print("👤 [async stream 第 3 轮]")
    print("🤖 ", end="", flush=True)
    async for chunk in llm.astream([
        HumanMessage(content="把我说的三个关键词用逗号连起来输出。"),
    ]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print("\n")

    print(f"✅ 异步调用同样保持会话记忆，共 {llm._session_turn} 轮\n")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

DEMOS = {
    "1": ("单实例多轮 — 自动会话记忆", demo_single_instance_multiturn),
    "2": ("会话重置 — 清空记忆", demo_session_reset),
    "3": ("并行独立会话 — 记忆隔离", demo_parallel_sessions),
    "4": ("LangGraph 集成 — 跨节点共享记忆", demo_langgraph_memory),
    "5": ("异步多轮 — 会话记忆", lambda: asyncio.run(demo_async_memory())),
}


def run_all():
    """依次运行所有演示（4 除外，因为它有代码审查逻辑）"""
    for k in ["1", "2", "3", "4", "5"]:
        name, fn = DEMOS[k]
        print(f"\n{'#'*60}")
        print(f"# 运行演示 {k}: {name}")
        print(f"{'#'*60}\n")
        fn()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in DEMOS:
        name, fn = DEMOS[sys.argv[1]]
        print(f"运行演示 {sys.argv[1]}: {name}\n")
        fn()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        run_all()
    else:
        print("用法: python demo_memory_agent.py [1-5|all]\n")
        for k, (name, _) in DEMOS.items():
            print(f"  {k}: {name}")
        print(f"\n默认运行演示 1（单实例多轮）...\n")
        demo_single_instance_multiturn()
