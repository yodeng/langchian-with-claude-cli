[English](./README.md) | [中文](./README_zh.md)

# langchain-with-claude-cli

将 [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) 包装为 LangChain 生态组件，提供三种后端集成方式加 Deep Agent 编排层，让 LangChain/LangGraph 应用获得 Claude Code 的完整文件系统、shell、git 操作能力。

## 目录

- [安装](#安装)
- [三种后端](#三种后端)
  - [后端 1: ChatModel — `ChatClaudeCode`](#后端-1-chatmodel--chatclaudecode)
  - [后端 2: Tool — `claude_code` 系列](#后端-2-tool--claude_code-系列)
  - [后端 3: Skill — `claude-code-cli`](#后端-3-skill--claude-code-cli)
- [Deep Agent 集成](#deep-agent-集成)
- [API 服务](#api-服务)
- [vs `deep_agent`](#vs-deep_agent)
- [完整工作流示例](#完整工作流示例)
- [配置参数参考](#配置参数参考)
- [项目结构](#项目结构)
- [开发指南](#开发指南)
- [许可证](#许可证)

## 安装

### 前置条件

- Python >= 3.10
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) >= 2.1.162
- `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 环境变量（`ChatOpenAI` 后端需要）
- `OPENAI_MODEL` 环境变量（可选，默认 `deepseek-v4-pro`）

### 安装项目

```bash
pip install -e .

# 使用 Agent 功能
pip install -e ".[agents]"

# 使用 Deep Agent 集成
pip install -e ".[deepagent]"

# 运行示例
pip install -e ".[examples]"

# 作为 API 服务运行
pip install -e ".[api]"
```

核心依赖：`langchain-core >= 1.0`、`pydantic >= 2.0`。

## 三种后端

### 后端 1: ChatModel — `ChatClaudeCode`

继承 `BaseChatModel`，与 `ChatOpenAI` 接口完全兼容，适合作为 LangGraph Agent 的 LLM 节点。

**特点：**
- 底层通过 `claude -p` 子进程调用，拥有完整文件系统、shell、git 能力
- `_generate` / `_stream` / `_agenerate` / `_astream` 全部实现
- 同一实例内自动通过 `--session-id` / `--resume` 保持多轮上下文
- `bind_tools()` 注入工具描述到 system prompt，Claude Code 原生执行
- `with_structured_output()` 通过 `--json-schema` 约束输出
- `reset_session()` 清空记忆，开始新对话

#### 基础用法

```python
from chat_claude_code import ChatClaudeCode

llm = ChatClaudeCode(working_dir="/path/to/project", effort="medium")

# 单轮调用
response = llm.invoke("分析这个项目的架构")
print(response.content)

# 多轮对话（自动保持上下文）
llm.invoke("列出所有 Python 文件")
llm.invoke("分析第一个文件的功能")  # 知道上一轮的上下文

# 流式输出
for chunk in llm.stream("生成一个排序算法"):
    print(chunk.content, end="", flush=True)

# 异步流式
async for chunk in llm.astream("优化这段代码"):
    print(chunk.content, end="", flush=True)
```

#### 结构化输出

```python
from pydantic import BaseModel

class FileInfo(BaseModel):
    path: str
    functions: list[str]
    classes: list[str]
    imports: list[str]

llm = ChatClaudeCode(working_dir="./my-project")
structured_llm = llm.with_structured_output(FileInfo)
result = structured_llm.invoke("分析 app.py 的结构")
# result 是 FileInfo 实例
```

#### 工具绑定

```python
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """读取文件内容"""
    with open(path) as f:
        return f.read()

llm = ChatClaudeCode(working_dir=".")
llm_with_tools = llm.bind_tools([read_file])
response = llm_with_tools.invoke("读取 README.md 的内容")
```

#### 在 LangGraph 中使用

```python
from langgraph.graph import StateGraph, END
from chat_claude_code import ChatClaudeCode

graph = StateGraph(AgentState)
graph.add_node("agent", lambda s: {
    "messages": [ChatClaudeCode(working_dir=".").invoke(s["messages"])]
})
graph.set_entry_point("agent")
graph.add_edge("agent", END)
```

---

### 后端 2: Tool — `claude_code` 系列

`@tool` 装饰的独立工具函数，适合让 LangGraph Agent 在特定步骤上将任务委托给 Claude Code 执行。

提供 4 个工具函数：

| 函数 | 用途 | 输出格式 |
|------|------|----------|
| `claude_code` | 基础委托 | 文本字符串 |
| `claude_code_structured` | 结构化输出 | JSON（`--json-schema` 约束） |
| `claude_code_isolated` | 隔离执行 | 文本字符串（git worktree 隔离） |
| `claude_code_streaming` | 流式委托 | `subprocess.Popen` 对象 |

#### 基础委托

```python
from claude_code_tool import claude_code

result = claude_code.invoke({
    "task": "扫描项目，列出所有函数签名",
    "effort": "low",
})
print(result)
```

#### 结构化输出

```python
from claude_code_tool import claude_code_structured

schema = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}},
        "total_functions": {"type": "integer"},
    },
    "required": ["files", "total_functions"],
}

result = claude_code_structured.invoke({
    "task": "分析项目结构，列出所有 Python 文件和函数总数",
    "output_schema": schema,
})
# result 是符合 schema 的 JSON 字符串
```

#### 隔离执行

```python
from claude_code_tool import claude_code_isolated

result = claude_code_isolated.invoke({
    "task": "修复 app.py 中的类型注解问题",
    "context_files": ["app.py"],
})
# 在独立的 git worktree 中执行，不影响主工作区
```

#### 多步会话委托

```python
from claude_code_tool import claude_code, close_session

SESSION = "code-review-001"

# 步骤 1: 探索代码
r1 = claude_code.invoke({
    "task": "分析项目结构，列出所有模块",
    "session_id": SESSION,
})

# 步骤 2: 在同一个 session 中继续（有上下文记忆）
r2 = claude_code.invoke({
    "task": "基于前面的分析，检查模块之间的循环依赖",
    "session_id": SESSION,
})

# 清理
close_session(SESSION)
```

#### 在 LangGraph Agent 中使用

```python
from langgraph.prebuilt import ToolNode
from claude_code_tool import claude_code, claude_code_structured

tools = [claude_code, claude_code_structured]
llm_with_tools = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "deepseek-v4-pro")).bind_tools(tools)

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
```

---

### 后端 3: Skill — `claude-code-cli`

Claude Code 自身的 skill 系统集成。`.claude/skills/claude-code-cli/` 让 Claude Code 以子进程方式调度另一个 Claude Code 实例，适用于：

- 需要子进程隔离的任务（独立 session、独立 worktree）
- 多级调用场景（外层 Claude Code 调度内层 Claude Code）
- 通过符号链接复用 `chat_claude_code.py` 和 `claude_code_tool.py`

```python
import os, sys
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), ".claude/skills/claude-code-cli"
))
from chat_claude_code import ChatClaudeCode

llm = ChatClaudeCode(working_dir="<目标目录>", effort="medium")
result = llm.invoke("分析这个项目的架构")
```

---

## Deep Agent 集成

`deep_agent` 模块将 LangChain 的 [`deep_agent`](https://docs.langchain.com/oss/python/deepagents/quickstart) 编排框架与 Claude Code CLI 执行能力结合。`deep_agent` 负责任务规划、拆解、子 agent 调度；Claude Code 负责文件操作、代码分析、shell 命令。

> **模型分离说明：** 默认模型 `"deepseek-v4-pro"` 仅用于 LangChain 编排层（规划、推理），Claude Code CLI 执行层使用自身配置的模型，两者相互独立。修改编排层模型不会影响 Claude Code 执行文件操作和代码分析时使用的模型。

### 预设模式

| 模式 | 说明 | Claude Code 工具 |
|------|------|-------------------|
| `code_analysis` | 只读分析（默认） | `claude_code`, `claude_code_structured` |
| `code_refactor` | 读写 + git 操作 | `claude_code`, `claude_code_structured` |
| `full_access` | 无限制 + 隔离执行 | `claude_code`, `claude_code_structured`, `claude_code_isolated` |
| `none` | 纯 deep_agent，不注册 Claude Code 工具 | — |

### 用法

```python
from deep_agent import create_claude_deep_agent

# 模式 A：标准 LLM 编排 + Claude Code 工具执行（推荐）
agent = create_claude_deep_agent(model="deepseek-v4-pro", mode="code_analysis")
result = agent.invoke({
    "messages": [{"role": "user", "content": "分析项目架构"}]
})

# 模式 B：ChatClaudeCode 作为 LLM 后端
from chat_claude_code import ChatClaudeCode
agent = create_claude_deep_agent(
    model=ChatClaudeCode(working_dir=".", effort="high"),
    mode="none",
)

# 快捷函数
from deep_agent import create_code_analysis_agent, create_code_refactor_agent
agent = create_code_refactor_agent(model="deepseek-v4-pro")

# 流式调用
for chunk, meta in agent.stream({"messages": [...]}, stream_mode="messages"):
    print(chunk.content, end="", flush=True)

# 自定义配置
agent = create_claude_deep_agent(
    model="deepseek-v4-pro",
    mode="code_refactor",
    claude_tool_effort="high",
    claude_tool_allowed=["Read", "Write", "Edit", "Bash(git *)"],
)
```

安装 `deepagent` extra：

```bash
pip install -e ".[deepagent]"
```

函数：`create_claude_deep_agent()`、`create_code_analysis_agent()`、`create_code_refactor_agent()`、`create_full_access_agent()`

---

## API 服务

将 `ChatClaudeCode` 和 `deep_agent` 暴露为 REST API 端点。

### 快速启动

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
# 或启用自动重载：
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

打开 http://localhost:8000/docs 查看交互式 Swagger 文档。

### 端点

**ChatClaudeCode** — 简单 LLM 对话：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 非流式对话 |
| `POST` | `/chat/stream` | 流式对话（SSE）|

**Deep Agent** — 编排执行，含任务规划 + 工具调用：

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/deep-agent` | 非流式 Deep Agent |
| `POST` | `/deep-agent/stream` | 流式 Deep Agent（SSE）|

**管理：**

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `/sessions` | 列出 `/chat` 活跃会话 |
| `DELETE` | `/sessions/{id}` | 删除 `/chat` 会话 |

### 示例

**ChatClaudeCode：**
```bash
# 非流式调用
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "分析项目结构", "working_dir": "."}'

# 流式调用（SSE）
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "解释这个代码库"}' \
  --no-buffer

# 多轮对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "列出所有 Python 文件", "session_id": "my-session"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "检查有没有错误", "session_id": "my-session"}'
```

**Deep Agent：**
```bash
# 非流式 — 代码分析（默认模式）
curl -X POST http://localhost:8000/deep-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "分析项目架构"}'

# 流式 — 代码重构模式
curl -X POST http://localhost:8000/deep-agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "重构 src/ 下的代码", "mode": "code_refactor"}' \
  --no-buffer

# 自定义配置
curl -X POST http://localhost:8000/deep-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "全面审查代码质量", "mode": "full_access", "claude_tool_effort": "high"}'
```

SSE 事件格式（`/chat/stream` 和 `/deep-agent/stream` 统一）：
```
data: {"type": "delta", "content": "你好"}
data: {"type": "delta", "content": "，这是分析结果"}
data: {"type": "done"}
```

---

## vs `deep_agent`

`langchain-with-claude-cli` 和 LangChain 的 [`deep_agent`](https://docs.langchain.com/oss/python/deepagents/quickstart) 都能构建强大的 Agent，但定位完全不同：

| 维度 | `deep_agent` | `ChatClaudeCode` |
|------|-------------|-------------------|
| **执行能力** | LLM 推理 + LangChain 工具调用 | 真实文件系统、shell、git 操作（通过 `claude` CLI） |
| **上下文管理** | LangChain 消息历史 + 中间件 hooks | Claude Code 原生 `--session-id` / `--resume` 持久化 |
| **工具生态** | LangChain 工具生态（Tavily、代码执行等） | Claude Code 完整工具集（Read、Write、Edit、Bash、Glob、Grep 等） |
| **子 Agent 模型** | 派生子 Agent，隔离上下文窗口 | `claude -p` 子进程隔离，独立 session |
| **文件系统访问** | 限于配置的工作目录 + 工具授权 | 完整项目级访问，支持 `--add-dir` 和 git worktree |
| **适合场景** | 多步推理、研究、规划、联网任务 | 代码分析、跨文件重构、文件操作、shell 自动化 |
| **集成方式** | 高层 Agent 框架，内置中间件 | 底层 `BaseChatModel` / `@tool`，可插入任意 LangChain 流水线 |

**如何选择：**

- 用 **`deep_agent`** — 推理密集型任务：研究综述、多跳 Q&A、网页抓取、带规划/todo 的 Agent 编排。
- 用 **`ChatClaudeCode`** — 需要真实代码执行：分析代码库、跨文件重构、执行 shell 命令、管理 git 工作流。
- **组合使用** — 用 `create_claude_deep_agent()`（见上方 [Deep Agent 集成](#deep-agent-集成)）一行代码同时获得 `deep_agent` 编排和 Claude Code 执行能力。

---

## 完整工作流示例

### 代码分析流水线 (`agents/code_analysis_agent.py`)

三阶段 LangGraph 流水线，使用 `ChatClaudeCode` 作为每个阶段的推理后端：

```
项目结构分析 → 代码质量分析 → 生成报告
```

```bash
python agents/code_analysis_agent.py /path/to/project
```

### 多轮记忆演示 (`agents/demo_memory_agent.py`)

展示 `ChatClaudeCode` 的会话记忆能力：

- **演示 1**: 单实例多轮对话，自动保持上下文
- **演示 2**: 主动 `reset_session()` 清空记忆
- **演示 3**: 两个独立实例，会话互不干扰
- **演示 4**: 跨 Graph 调用，记忆持久化

```bash
python agents/demo_memory_agent.py 1   # 单实例多轮
python agents/demo_memory_agent.py 2   # 会话重置
```

### Tool 委托示例 (`examples/example_tool.py`)

四种委托模式：

| 模式 | 演示函数 |
|------|----------|
| 基础委托 | `example_basic_delegation()` |
| 会话委托 | `example_session_delegation()` |
| 结构化委托 | `example_structured_delegation()` |
| 完整 Agent | `example_full_agent()` |

```bash
python examples/example_tool.py 1   # 基础委托
python examples/example_tool.py 4   # 完整 Agent
```

### ChatModel 示例 (`examples/example_chat_model.py`)

7 种使用模式，覆盖 invoke/stream/async/tools/structured 全部场景。

```bash
python examples/example_chat_model.py
```

---

## 配置参数参考

### ChatClaudeCode 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `working_dir` | `str` | `"."` | 工作目录，Claude Code 在此目录下执行 |
| `effort` | `str` | `"medium"` | 努力级别：`low` / `medium` / `high` / `xhigh` / `max` |
| `model` | `str \| None` | `None` | 模型名称（默认使用 Claude Code 配置） |
| `max_tokens` | `int \| None` | `None` | 最大输出 token 数 |
| `timeout` | `int` | `600` | 超时秒数（10–3600） |
| `skip_permissions` | `bool` | `True` | 跳过权限确认（自动化场景推荐开启） |
| `allowed_tools` | `list[str] \| None` | `None` | 允许的工具列表，如 `["Read", "Write"]` |
| `disallowed_tools` | `list[str] \| None` | `None` | 禁止的工具列表 |
| `system_prompt` | `str \| None` | `None` | 自定义系统提示词 |
| `context_files` | `list[str] \| None` | `None` | 授权访问的文件/目录 |
| `extra_env` | `dict[str, str]` | `{}` | 额外环境变量 |

### claude_code 系列参数

| 参数 | 说明 |
|------|------|
| `task` | 详细的任务描述 |
| `session_id` | 会话 ID，空字符串 = 独立调用，相同 ID = 延续上下文 |
| `allowed_tools` | 允许的工具列表（仅 `claude_code`） |
| `output_schema` | JSON Schema 定义（仅 `claude_code_structured`） |
| `context_files` | 授权文件列表（仅 `claude_code_isolated`） |
| `effort` | 努力级别：`low` / `medium` / `high` |

### create_claude_deep_agent 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str \| BaseChatModel \| None` | `"deepseek-v4-pro"` | LLM 后端，支持字符串或 `ChatClaudeCode` 实例 |
| `mode` | `str` | `"code_analysis"` | 预设模式：`code_analysis` / `code_refactor` / `full_access` / `none` |
| `tools` | `Sequence \| None` | `None` | 额外的 LangChain 工具，与 Claude Code 工具合并 |
| `claude_tool_effort` | `str \| None` | 按 mode | 覆盖 effort 级别：`low` / `medium` / `high` / `xhigh` / `max` |
| `claude_tool_allowed` | `list[str] \| None` | 按 mode | 覆盖 `claude_code` 的允许工具列表 |
| `system_prompt` | `str \| None` | 自动生成 | 自定义系统提示词 |
| `permissions` | `list \| None` | `None` | 文件系统权限规则 |
| `backend` | `Any \| None` | `None` | deep_agent 后端（存储/沙箱） |
| `subagents` | `Sequence \| None` | `None` | 额外的子 agent 定义 |
| `skills` | `list[str] \| None` | `None` | 技能文件路径 |
| `memory` | `list[str] \| None` | `None` | 记忆文件路径 |

---

## 项目结构

```
langchain-with-claude-cli/
├── chat_claude_code.py         # 后端 1: ChatClaudeCode — BaseChatModel 子类
├── claude_code_tool.py         # 后端 2: claude_code 系列 — @tool 封装
├── deep_agent.py               # Deep Agent: deep_agent 编排 + Claude Code 工具
├── pyproject.toml              # 项目配置与依赖声明
├── agents/                     # LangGraph Agent 实现
│   ├── code_analysis_agent.py  #   三阶段代码分析流水线
│   └── demo_memory_agent.py    #   多轮对话记忆演示
├── examples/                   # 独立示例脚本
│   ├── example_chat_model.py   #   ChatClaudeCode 7 种使用模式
│   └── example_tool.py         #   Tool 委托 4 种用法
├── tests/                      # 测试套件（141 个用例）
│   ├── test_chat_claude_code.py
│   └── test_claude_code_tool.py
├── .claude/skills/             # 后端 3: Claude Code skill 系统
│   └── claude-code-cli/        #   skill 定义 + 符号链接
└── api_server.py               # FastAPI 服务（流式 + 非流式）
```

---

## 开发指南

### 技术要点

- 依赖 `claude` CLI >= 2.1.162，`langchain-core >= 1.0`
- 流式输出需要 `--verbose` 配合 `--output-format stream-json`
- 事件格式：外层 `stream_event` 包装内层 `event`，需解包后解析 `content_block_delta`
- `usage_metadata` 必须包含 `total_tokens` 字段

### 提交规范

- 英文一句话，小写开头，不加 feat/fix 前缀
- 不需要 `Co-Authored-By`
- 参考 `.claude/COMMIT_CONVENTION.md`

### 添加新工具函数

在 `claude_code_tool.py` 中：

```python
@tool
def my_new_tool(task: str, ...) -> str:
    """工具描述"""
    result = _run_claude_code(task, ...)
    return result.output
```

遵循 `claude_code` 的命名风格（不加多余前缀），保持与现有工具一致。

---

## 许可证

MIT
