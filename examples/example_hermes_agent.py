"""
Hermes Agent 集成使用示例

演示 create_hermes_agent() 的 4 种使用模式：

1. 代码分析模式（code_analysis）
2. 代码重构模式（code_refactor）
3. 完全访问模式（full_access）
4. 纯 Hermes Agent 模式（none）

启动方式:
    python examples/example_hermes_agent.py

前置条件:
    - Hermes Agent 已安装并配置
    - deepagents 已安装: pip install deepagents
    - Hermes Agent API 已启动（默认 http://localhost:30000/v1）
    - claude CLI 已安装（Claude Code 工具需要）
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def demo_1_code_analysis():
    """示例 1：代码分析模式 — 只读分析"""
    print("=" * 60)
    print("示例 1：代码分析模式（code_analysis）")
    print("=" * 60)

    from hermes_agent import create_hermes_agent

    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="code_analysis",
        working_dir=".",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "分析当前项目目录结构，列出所有 Python 文件并描述项目的主要模块。",
        }]
    })
    final_msg = result["messages"][-1]
    print(f"回复: {final_msg.content[:500]}...")
    print()


def demo_2_code_refactor():
    """示例 2：代码重构模式 — 文件读写 + git"""
    print("=" * 60)
    print("示例 2：代码重构模式（code_refactor）")
    print("=" * 60)

    from hermes_agent import create_hermes_agent

    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="code_refactor",
        working_dir=".",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "检查项目中是否有 TODO 或 FIXME 注释，列出它们的位置和内容。",
        }]
    })
    final_msg = result["messages"][-1]
    print(f"回复: {final_msg.content[:500]}...")
    print()


def demo_3_full_access():
    """示例 3：完全访问模式 — 无限制 + 隔离执行"""
    print("=" * 60)
    print("示例 3：完全访问模式（full_access）")
    print("=" * 60)

    from hermes_agent import create_hermes_agent

    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="full_access",
        working_dir=".",
        claude_tool_effort="high",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "运行 python 项目的测试套件并报告结果。项目在 examples/ 目录下的测试文件。",
        }]
    })
    final_msg = result["messages"][-1]
    print(f"回复: {final_msg.content[:500]}...")
    print()


def demo_4_pure_hermes():
    """示例 4：纯 Hermes Agent — 不注册 Claude Code 工具"""
    print("=" * 60)
    print("示例 4：纯 Hermes Agent 模式（none）")
    print("=" * 60)

    from hermes_agent import create_hermes_agent

    agent = create_hermes_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        mode="none",
        working_dir=".",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "作为一个代码 Agent，简述你的能力范围。你能执行哪些类型的任务？",
        }]
    })
    final_msg = result["messages"][-1]
    print(f"回复: {final_msg.content[:500]}...")
    print()


def demo_5_shortcut_functions():
    """示例 5：快捷函数"""
    print("=" * 60)
    print("示例 5：快捷函数")
    print("=" * 60)

    from hermes_agent import (
        create_hermes_code_analysis_agent,
        create_hermes_code_refactor_agent,
        create_hermes_full_access_agent,
    )

    # 使用快捷函数
    agent = create_hermes_code_analysis_agent(
        base_url="http://localhost:30000/v1",
        hermes_model="claude-opus-4-20250514",
        working_dir=".",
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "项目中用了哪些第三方依赖？列出它们。"}]
    })
    final_msg = result["messages"][-1]
    print(f"快捷函数回复: {final_msg.content[:300]}...")
    print()


def demo_6_with_chat_hermes_agent():
    """示例 6：传入已有的 ChatHermesAgent 实例"""
    print("=" * 60)
    print("示例 6：传入 ChatHermesAgent 实例")
    print("=" * 60)

    from chat_hermes_agent import ChatHermesAgent
    from hermes_agent import create_hermes_agent

    llm = ChatHermesAgent(
        base_url="http://localhost:30000/v1",
        model="claude-opus-4-20250514",
        working_dir=".",
        system_prompt="你是一个专注于代码质量的资深工程师。",
    )

    agent = create_hermes_agent(
        model=llm,
        mode="code_analysis",
        working_dir=".",
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "给出 3 条提高代码可维护性的建议。",
        }]
    })
    final_msg = result["messages"][-1]
    print(f"回复: {final_msg.content[:500]}...")
    print()


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Hermes Agent 集成使用示例")
    print("前置条件:")
    print("  1. Hermes Agent API 运行在 http://localhost:30000/v1")
    print("  2. deepagents 已安装: pip install deepagents")
    print("  3. claude CLI 已安装（使用 code_analysis/refactor/full_access 模式时需要）")
    print()

    examples = [
        ("1", "代码分析模式", demo_1_code_analysis),
        ("2", "代码重构模式", demo_2_code_refactor),
        ("3", "完全访问模式", demo_3_full_access),
        ("4", "纯 Hermes Agent", demo_4_pure_hermes),
        ("5", "快捷函数", demo_5_shortcut_functions),
        ("6", "传入 ChatHermesAgent 实例", demo_6_with_chat_hermes_agent),
    ]

    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print()

    choice = input("选择要运行的示例 (1-6, 或 'all' 全部运行): ").strip()

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
