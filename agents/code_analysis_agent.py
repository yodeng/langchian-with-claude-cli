"""
LangGraph 代码分析 Agent — 使用 ChatClaudeCode 作为后端

三阶段流水线：
  1. analyze_structure — 项目结构分析（文件清单、模块依赖、目录树）
  2. analyze_quality   — 代码质量分析（函数长度、复杂度、注释率、命名规范等）
  3. generate_report    — 汇总生成 Markdown 报告并写入文件

运行：
  python code_analysis_agent.py [项目路径]

默认分析当前项目目录。
"""

import os
import sys
import uuid
from typing import Annotated, TypedDict

# 确保能导入父目录的 chat_claude_code
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chat_claude_code import ChatClaudeCode


# ═══════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════

class CodeAnalysisState(TypedDict):
    """代码分析 Agent 的全局状态"""

    messages: Annotated[list[BaseMessage], add_messages]
    project_path: str
    structure_data: str
    quality_data: str
    report_path: str


# ═══════════════════════════════════════════════════════════════
# 节点 1: 项目结构分析
# ═══════════════════════════════════════════════════════════════

def analyze_structure(state: CodeAnalysisState) -> dict:
    """分析项目结构：文件清单、模块依赖、目录树"""
    project = state["project_path"]
    project_name = os.path.basename(project.rstrip("/"))

    llm = ChatClaudeCode(
        working_dir=project,
        effort="high",
        timeout=180,
        allowed_tools=["Read", "Bash", "Glob"],
        system_prompt=(
            f"你是项目结构分析师。你的唯一任务是对 {project_name} "
            f"进行结构分析，输出结构化的 Markdown 报告。\n\n"
            "## 必须包含的内容\n\n"
            "### 1. 项目概览\n"
            "- 目录深度、总 Python 文件数、总代码行数（用 Bash 工具统计）\n\n"
            "### 2. 文件清单\n"
            "- 按模块/目录分组，列出所有 .py 文件\n"
            "- 标注每个文件的代码行数和大致功能（只读前 20 行即可）\n\n"
            "### 3. 模块依赖关系\n"
            "- 识别每个文件 import 了哪些内部模块\n"
            "- 用文本箭头表示依赖方向：`模块A → 模块B`\n\n"
            "### 4. 入口识别\n"
            "- 找到 main 入口、CLI 入口\n"
            "- 标注 if __name__ == \"__main__\" 位置\n\n"
            "### 5. 目录结构树\n"
            "- 只显示到 2 层深度\n\n"
            "## 约束\n"
            "- 不要做代码质量分析（那是下一步的工作）\n"
            "- 不要修改任何文件，只读取和分析\n"
            "- 用中文输出"
        ),
    )

    response = llm.invoke([
        HumanMessage(content=f"请分析项目 {project} 的完整结构，按上述要求输出结构化 Markdown 报告。")
    ])

    return {
        "messages": [response],
        "structure_data": response.content,
    }


# ═══════════════════════════════════════════════════════════════
# 节点 2: 代码质量分析
# ═══════════════════════════════════════════════════════════════

def analyze_quality(state: CodeAnalysisState) -> dict:
    """分析代码质量：函数长度、复杂度、注释率、命名规范等"""
    project = state["project_path"]
    structure = state.get("structure_data", "(结构分析不可用)")

    llm = ChatClaudeCode(
        working_dir=project,
        effort="high",
        timeout=300,
        allowed_tools=["Read", "Bash", "Grep"],
        system_prompt=(
            "你是代码质量分析师。基于项目结构分析结果，对每个 Python 源文件"
            "从以下维度进行定量质量评估，输出结构化 Markdown 报告。\n\n"
            "## 分析维度\n\n"
            "### 1. 函数/方法长度分布\n"
            "- 列出超过 50 行的函数，标注文件名、行号、行数\n"
            "- 统计：最长函数行数、平均函数行数\n\n"
            "### 2. 注释覆盖率\n"
            "- 对每个文件统计注释行占比 = 注释行数/总行数×100%\n"
            "- 标注低于 10% 的文件\n"
            "- 计算项目级注释覆盖率均值\n\n"
            "### 3. 命名规范\n"
            "- 检测不符合 Python snake_case 命名的函数名/变量名\n"
            "- 检测不符合 PascalCase 的类名\n\n"
            "### 4. 异常处理\n"
            "- 检测 bare except (except:) 语句\n"
            "- 检测 except pass 静默吞异常\n\n"
            "### 5. 代码重复\n"
            "- 识别跨文件的明显重复代码块\n"
            "- 列出重复的 import 组、重复的工具函数\n\n"
            "### 6. 综合评分\n"
            "- 对每个维度给出 1-10 分\n"
            "- 计算综合得分\n\n"
            "## 约束\n"
            "- 跳过 test_ 开头的测试文件\n"
            "- 跳过 __pycache__ 目录\n"
            "- 只分析 .py 文件\n"
            "- 所有指标给出定量数值\n"
            "- 不要修改任何文件\n"
            "- 用中文输出"
        ),
    )

    response = llm.invoke([
        HumanMessage(
            content=(
                "## 项目结构分析（供参考）\n\n"
                f"{structure}\n\n"
                "---\n\n"
                f"请基于以上信息，对 {project} 的源代码进行深入的代码质量分析。"
            )
        )
    ])

    return {
        "messages": [response],
        "quality_data": response.content,
    }


# ═══════════════════════════════════════════════════════════════
# 节点 3: 生成报告
# ═══════════════════════════════════════════════════════════════

def generate_report(state: CodeAnalysisState) -> dict:
    """汇总两份分析结果，生成 Markdown 报告文件"""
    project = state["project_path"]
    project_name = os.path.basename(project.rstrip("/"))
    structure = state.get("structure_data", "")
    quality = state.get("quality_data", "")

    report_filename = f"code_analysis_report_{uuid.uuid4().hex[:8]}.md"
    report_path = os.path.join(project, report_filename)

    llm = ChatClaudeCode(
        working_dir=project,
        effort="medium",
        timeout=120,
        allowed_tools=["Write"],
        system_prompt=(
            f"你是报告生成器。将结构分析和质量分析的结果合并为一份排版精美的 Markdown 报告。\n\n"
            "## 报告结构\n\n"
            "```\n"
            "# 代码分析报告 — {project_name}\n\n"
            "## 📊 分析概览（关键指标表格）\n"
            "## 🏗 项目结构\n"
            "## 🔍 代码质量详情\n"
            "## 🎯 雷达评分（各维度 1-10 分）\n"
            "## 💡 改进建议（Top-5 优先级排序）\n"
            "```\n\n"
            "## 约束\n"
            "- 不要重新做分析，只汇总已有的两份数据\n"
            "- 分析概览用表格汇总所有可量化的指标\n"
            "- 改进建议要具体可执行，按优先级排序\n"
            "- 用 Write 工具写入文件，路径: {report_path}\n"
            "- 写入成功后只输出报告路径\n"
            "- 用中文撰写\n"
        ),
    )

    response = llm.invoke([
        HumanMessage(
            content=(
                "## 结构分析\n\n"
                f"{structure}\n\n"
                "---\n\n"
                "## 代码质量分析\n\n"
                f"{quality}\n\n"
                "---\n\n"
                f"请合并以上两份报告，写入 {report_path}。"
            )
        )
    ])

    return {
        "messages": [response],
        "report_path": report_path,
    }


# ═══════════════════════════════════════════════════════════════
# 编译 Agent
# ═══════════════════════════════════════════════════════════════

def build_agent():
    """构建代码分析 Agent 图"""
    builder = StateGraph(CodeAnalysisState)

    builder.add_node("analyze_structure", analyze_structure)
    builder.add_node("analyze_quality", analyze_quality)
    builder.add_node("generate_report", generate_report)

    builder.set_entry_point("analyze_structure")
    builder.add_edge("analyze_structure", "analyze_quality")
    builder.add_edge("analyze_quality", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════
# 运行入口
# ═══════════════════════════════════════════════════════════════

def run_analysis(project_path: str) -> str:
    """运行完整的代码分析 pipeline，返回报告路径"""
    if not os.path.isdir(project_path):
        raise NotADirectoryError(f"项目路径不存在: {project_path}")

    abs_path = os.path.abspath(project_path)
    project_name = os.path.basename(abs_path.rstrip("/"))

    print(f"{'='*60}")
    print(f"🔬 代码分析 Agent 启动")
    print(f"📁 目标项目: {abs_path}")
    print(f"📋 流水线: 结构分析 → 质量分析 → 报告生成")
    print(f"{'='*60}\n")

    agent = build_agent()

    result = agent.invoke({
        "project_path": abs_path,
        "messages": [],
    })

    report_path = result.get("report_path", "")
    print(f"\n{'='*60}")
    print(f"✅ 分析完成！报告已生成: {report_path}")
    print(f"{'='*60}")
    return report_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "."

    run_analysis(target)
