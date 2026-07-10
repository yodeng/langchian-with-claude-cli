[English](./README.md) | [中文](./README_zh.md)

# langchain-with-claude-cli

Wrap [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) as LangChain ecosystem components, providing three backend integration approaches plus a deep agent orchestration layer. Give your LangChain/LangGraph applications full filesystem, shell, and git capabilities.

## Table of Contents

- [Installation](#installation)
- [Three Backends](#three-backends)
  - [Backend 1: ChatModel — `ChatClaudeCode`](#backend-1-chatmodel--chatclaudecode)
  - [Backend 2: Tool — `claude_code` Series](#backend-2-tool--claude_code-series)
  - [Backend 3: Skill — `claude-code-cli`](#backend-3-skill--claude-code-cli)
- [Deep Agent Integration](#deep-agent-integration)
- [API Server](#api-server)
- [vs `deep_agent`](#vs-deep_agent)
- [Complete Workflow Examples](#complete-workflow-examples)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)
- [License](#license)

## Installation

### Prerequisites

- Python >= 3.10
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) >= 2.1.162
- `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables (required for `ChatOpenAI` backend)
- `OPENAI_MODEL` environment variable (optional, defaults to `deepseek-v4-pro`)

### Install

```bash
pip install -e .

# With Agent support
pip install -e ".[agents]"

# With deep agent integration
pip install -e ".[deepagent]"

# With examples
pip install -e ".[examples]"

# Run as API server
pip install -e ".[api]"
```

Core dependencies: `langchain-core >= 1.0`, `pydantic >= 2.0`.

## Three Backends

### Backend 1: ChatModel — `ChatClaudeCode`

Extends `BaseChatModel` with full `ChatOpenAI` API compatibility. Ideal as the LLM node in a LangGraph agent.

**Highlights:**
- Invokes `claude -p` as a subprocess with full filesystem, shell, and git access
- Implements `_generate` / `_stream` / `_agenerate` / `_astream`
- Multi-turn context automatically maintained via `--session-id` / `--resume`
- `bind_tools()` injects tool descriptions into the system prompt; Claude Code executes tools natively
- `with_structured_output()` uses `--json-schema` to constrain output format
- `reset_session()` clears memory for a fresh conversation

#### Basic Usage

```python
from chat_claude_code import ChatClaudeCode

llm = ChatClaudeCode(working_dir="/path/to/project", effort="medium")

# Single-turn
response = llm.invoke("Analyze the project architecture")
print(response.content)

# Multi-turn (context is automatically preserved)
llm.invoke("List all Python files")
llm.invoke("Describe the first file")  # remembers the previous context

# Streaming
for chunk in llm.stream("Write a sorting algorithm"):
    print(chunk.content, end="", flush=True)

# Async streaming
async for chunk in llm.astream("Optimize this code"):
    print(chunk.content, end="", flush=True)
```

#### Structured Output

```python
from pydantic import BaseModel

class FileInfo(BaseModel):
    path: str
    functions: list[str]
    classes: list[str]
    imports: list[str]

llm = ChatClaudeCode(working_dir="./my-project")
structured_llm = llm.with_structured_output(FileInfo)
result = structured_llm.invoke("Analyze app.py structure")
# result is a FileInfo instance
```

#### Tool Binding

```python
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read a file's contents"""
    with open(path) as f:
        return f.read()

llm = ChatClaudeCode(working_dir=".")
llm_with_tools = llm.bind_tools([read_file])
response = llm_with_tools.invoke("Read README.md content")
```

#### Using in LangGraph

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

### Backend 2: Tool — `claude_code` Series

`@tool`-decorated functions that let a LangGraph agent delegate specific steps to Claude Code.

Four tool functions are provided:

| Function | Purpose | Output Format |
|----------|---------|---------------|
| `claude_code` | Basic delegation | Plain text |
| `claude_code_structured` | Structured output | JSON (constrained by `--json-schema`) |
| `claude_code_isolated` | Isolated execution | Plain text (git worktree isolation) |
| `claude_code_streaming` | Streaming delegation | `subprocess.Popen` object |

#### Basic Delegation

```python
from claude_code_tool import claude_code

result = claude_code.invoke({
    "task": "Scan the project and list all function signatures",
    "effort": "low",
})
print(result)
```

#### Structured Output

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
    "task": "Analyze the project structure — list all Python files and total function count",
    "output_schema": schema,
})
# result is a JSON string matching the schema
```

#### Isolated Execution

```python
from claude_code_tool import claude_code_isolated

result = claude_code_isolated.invoke({
    "task": "Fix type annotation issues in app.py",
    "context_files": ["app.py"],
})
# Runs in an independent git worktree; does not affect the main workspace
```

#### Multi-step Session Delegation

```python
from claude_code_tool import claude_code, close_session

SESSION = "code-review-001"

# Step 1: Explore
r1 = claude_code.invoke({
    "task": "Analyze the project structure and list all modules",
    "session_id": SESSION,
})

# Step 2: Continue in the same session (context is preserved)
r2 = claude_code.invoke({
    "task": "Based on the previous analysis, check for circular dependencies",
    "session_id": SESSION,
})

# Clean up
close_session(SESSION)
```

#### Using in a LangGraph Agent

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

### Backend 3: Skill — `claude-code-cli`

Integration with Claude Code's own skill system. `.claude/skills/claude-code-cli/` enables Claude Code to invoke another Claude Code instance as a subprocess. Suitable for:

- Isolated subprocess execution (independent session and worktree)
- Multi-level invocation (outer Claude Code orchestrates inner Claude Code)
- Reuses `chat_claude_code.py` and `claude_code_tool.py` via symlinks

```python
import os, sys
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), ".claude/skills/claude-code-cli"
))
from chat_claude_code import ChatClaudeCode

llm = ChatClaudeCode(working_dir="<target-dir>", effort="medium")
result = llm.invoke("Analyze the project architecture")
```

---

## Deep Agent Integration

The `deep_agent` module combines LangChain's [`deep_agent`](https://docs.langchain.com/oss/python/deepagents/quickstart) orchestration framework with Claude Code CLI execution. `deep_agent` handles planning, task decomposition, and sub-agent scheduling, while Claude Code handles file operations, code analysis, and shell commands.

> **Model separation:** The default model `"deepseek-v4-pro"` is used by the LangChain orchestration layer (planning, reasoning). The Claude Code CLI execution layer uses its own configured model — the two are independent. Changing the orchestration model does not affect which model Claude Code uses for file operations and code analysis.

### Preset Modes

| Mode | Description | Claude Code Tools |
|------|-------------|-------------------|
| `code_analysis` | Read-only analysis (default) | `claude_code`, `claude_code_structured` |
| `code_refactor` | Read/write + git operations | `claude_code`, `claude_code_structured` |
| `full_access` | Unrestricted + isolated execution | `claude_code`, `claude_code_structured`, `claude_code_isolated` |
| `none` | Pure deep_agent, no Claude Code tools | — |

### Usage

```python
from deep_agent import create_claude_deep_agent

# Pattern A: Standard LLM orchestrates, Claude Code executes (recommended)
agent = create_claude_deep_agent(model="deepseek-v4-pro", mode="code_analysis")
result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze the project architecture"}]
})

# Pattern B: ChatClaudeCode as the LLM backend
from chat_claude_code import ChatClaudeCode
agent = create_claude_deep_agent(
    model=ChatClaudeCode(working_dir=".", effort="high"),
    mode="none",
)

# Shortcut functions
from deep_agent import create_code_analysis_agent, create_code_refactor_agent
agent = create_code_refactor_agent(model="deepseek-v4-pro")

# Streaming
for chunk, meta in agent.stream({"messages": [...]}, stream_mode="messages"):
    print(chunk.content, end="", flush=True)

# Custom configuration
agent = create_claude_deep_agent(
    model="deepseek-v4-pro",
    mode="code_refactor",
    claude_tool_effort="high",
    claude_tool_allowed=["Read", "Write", "Edit", "Bash(git *)"],
)
```

Install with the `deepagent` extra:

```bash
pip install -e ".[deepagent]"
```

Functions: `create_claude_deep_agent()`, `create_code_analysis_agent()`, `create_code_refactor_agent()`, `create_full_access_agent()`

---

## API Server

Expose `ChatClaudeCode` and `deep_agent` as REST API endpoints.

### Quick Start

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
# or with auto-reload:
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### Endpoints

**ChatClaudeCode** — simple conversational LLM:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Non-streaming chat |
| `POST` | `/chat/stream` | Streaming chat (SSE) |

**Deep Agent** — orchestrated execution with planning + tools:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/deep-agent` | Non-streaming deep agent |
| `POST` | `/deep-agent/stream` | Streaming deep agent (SSE) |

**Management:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/sessions` | List active `/chat` sessions |
| `DELETE` | `/sessions/{id}` | Delete a `/chat` session |

### Examples

**ChatClaudeCode:**
```bash
# Non-streaming
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the project structure", "working_dir": "."}'

# Streaming (SSE)
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain this codebase"}' \
  --no-buffer

# Multi-turn conversation
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "list all Python files", "session_id": "my-session"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "now check them for errors", "session_id": "my-session"}'
```

**Deep Agent:**
```bash
# Non-streaming — code analysis (default mode)
curl -X POST http://localhost:8000/deep-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "Analyze the project architecture"}'

# Streaming — full access mode
curl -X POST http://localhost:8000/deep-agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "重构 src/ 下的代码", "mode": "code_refactor"}' \
  --no-buffer

# Custom mode
curl -X POST http://localhost:8000/deep-agent \
  -H "Content-Type: application/json" \
  -d '{"message": "全面审查代码质量", "mode": "full_access", "claude_tool_effort": "high"}'
```

SSE event format (same for both `/chat/stream` and `/deep-agent/stream`):
```
data: {"type": "delta", "content": "Hello"}
data: {"type": "delta", "content": " World"}
data: {"type": "done"}
```

---

## vs `deep_agent`

Both `langchain-with-claude-cli` and LangChain's [`deep_agent`](https://docs.langchain.com/oss/python/deepagents/quickstart) let you build powerful agents, but they serve fundamentally different needs:

| Dimension | `deep_agent` | `ChatClaudeCode` |
|-----------|-------------|-------------------|
| **Execution capability** | LLM reasoning + tool calling via LangChain tools | Real filesystem, shell, and git operations via `claude` CLI |
| **Context management** | LangChain-managed message history with middleware hooks | Claude Code native `--session-id` / `--resume` persistence |
| **Tool ecosystem** | LangChain tool ecosystem (Tavily, code execution, etc.) | Full Claude Code tool set (Read, Write, Edit, Bash, Glob, Grep, etc.) |
| **Sub-agent model** | Spawns sub-agents with isolated context windows | Subprocess isolation via `claude -p` with independent sessions |
| **Filesystem access** | Limited to configured working directory + tool grants | Full project-level access with `--add-dir` and git worktree support |
| **Best for** | Multi-step reasoning, research, planning, and web-connected tasks | Code analysis, project-wide refactoring, file operations, and shell automation |
| **Integration style** | High-level agent framework with built-in middleware | Low-level `BaseChatModel` / `@tool` — plugs into any LangChain pipeline |

**When to choose which:**

- Use **`deep_agent`** when your task is reasoning-heavy — research synthesis, multi-hop Q&A, web scraping, or agent orchestration with planning/todo lists.
- Use **`ChatClaudeCode`** when your task needs real code execution — analyzing a codebase, refactoring across files, running shell commands, or managing a git workflow.
- **Combine both**: use `create_claude_deep_agent()` (see [Deep Agent Integration](#deep-agent-integration) above) to get `deep_agent` orchestration with Claude Code execution in a single function call.

---

## Complete Workflow Examples

### Code Analysis Pipeline (`agents/code_analysis_agent.py`)

A three-stage LangGraph pipeline using `ChatClaudeCode` as the reasoning backend at each stage:

```
Structure Analysis → Quality Analysis → Report Generation
```

```bash
python agents/code_analysis_agent.py /path/to/project
```

### Multi-turn Memory Demo (`agents/demo_memory_agent.py`)

Demonstrates `ChatClaudeCode` session memory capabilities:

- **Demo 1**: Single-instance multi-turn with automatic context preservation
- **Demo 2**: Explicit `reset_session()` to clear memory
- **Demo 3**: Two independent instances with isolated sessions
- **Demo 4**: Cross-graph invocations with persistent memory

```bash
python agents/demo_memory_agent.py 1   # single-instance multi-turn
python agents/demo_memory_agent.py 2   # session reset
```

### Tool Delegation Examples (`examples/example_tool.py`)

Four delegation patterns:

| Pattern | Function |
|---------|----------|
| Basic delegation | `example_basic_delegation()` |
| Session delegation | `example_session_delegation()` |
| Structured delegation | `example_structured_delegation()` |
| Full agent | `example_full_agent()` |

```bash
python examples/example_tool.py 1   # basic delegation
python examples/example_tool.py 4   # full agent
```

### ChatModel Examples (`examples/example_chat_model.py`)

Seven usage patterns covering invoke, stream, async, tools, and structured output.

```bash
python examples/example_chat_model.py
```

---

## Configuration Reference

### ChatClaudeCode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `working_dir` | `str` | `"."` | Working directory for Claude Code execution |
| `effort` | `str` | `"medium"` | Effort level: `low` / `medium` / `high` / `xhigh` / `max` |
| `model` | `str \| None` | `None` | Model name (uses Claude Code default if unset) |
| `max_tokens` | `int \| None` | `None` | Maximum output tokens |
| `timeout` | `int` | `600` | Timeout in seconds (10–3600) |
| `skip_permissions` | `bool` | `True` | Skip permission prompts (recommended for automation) |
| `allowed_tools` | `list[str] \| None` | `None` | Allowed tools, e.g. `["Read", "Write"]` |
| `disallowed_tools` | `list[str] \| None` | `None` | Disallowed tools |
| `system_prompt` | `str \| None` | `None` | Custom system prompt |
| `context_files` | `list[str] \| None` | `None` | Additional files/directories to grant access to |
| `extra_env` | `dict[str, str]` | `{}` | Extra environment variables |

### claude_code Series Parameters

| Parameter | Applies To | Description |
|-----------|------------|-------------|
| `task` | All | Detailed task description |
| `session_id` | `claude_code`, `claude_code_structured` | Session ID: empty = isolated, same ID = shared context |
| `allowed_tools` | `claude_code` | Allowed tool list |
| `output_schema` | `claude_code_structured` | JSON Schema for output constraints |
| `context_files` | `claude_code_isolated` | Authorized file/directory paths |
| `effort` | All (except streaming) | Effort level: `low` / `medium` / `high` |

### create_claude_deep_agent Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| BaseChatModel \| None` | `"deepseek-v4-pro"` | LLM backend — string, `ChatClaudeCode`, or any `BaseChatModel` |
| `mode` | `str` | `"code_analysis"` | Preset mode: `code_analysis` / `code_refactor` / `full_access` / `none` |
| `tools` | `Sequence \| None` | `None` | Additional LangChain tools merged with Claude Code tools |
| `claude_tool_effort` | `str \| None` | per mode | Override effort: `low` / `medium` / `high` / `xhigh` / `max` |
| `claude_tool_allowed` | `list[str] \| None` | per mode | Override allowed tools for `claude_code` |
| `system_prompt` | `str \| None` | auto-generated | Custom system prompt |
| `permissions` | `list \| None` | `None` | Filesystem permission rules |
| `backend` | `Any \| None` | `None` | deep_agent backend (storage/sandbox) |
| `subagents` | `Sequence \| None` | `None` | Additional sub-agent specs |
| `skills` | `list[str] \| None` | `None` | Skill source paths |
| `memory` | `list[str] \| None` | `None` | Memory file paths |

---

## Project Structure

```
langchain-with-claude-cli/
├── chat_claude_code.py         # Backend 1: ChatClaudeCode — BaseChatModel subclass
├── claude_code_tool.py         # Backend 2: claude_code series — @tool wrappers
├── deep_agent.py               # Deep Agent: deep_agent orchestration + Claude Code tools
├── pyproject.toml              # Project config & dependencies
├── agents/                     # LangGraph agent implementations
│   ├── code_analysis_agent.py  #   Three-stage code analysis pipeline
│   └── demo_memory_agent.py    #   Multi-turn memory demo
├── examples/                   # Standalone example scripts
│   ├── example_chat_model.py   #   ChatClaudeCode: 7 usage patterns
│   └── example_tool.py         #   Tool delegation: 4 patterns
├── tests/                      # Test suite (141 cases)
│   ├── test_chat_claude_code.py
│   └── test_claude_code_tool.py
├── .claude/skills/             # Backend 3: Claude Code skill system
│   └── claude-code-cli/        #   Skill definition + symlinks
└── api_server.py               # FastAPI server (streaming + non-streaming)
```

---

## Development Guide

### Technical Notes

- Requires `claude` CLI >= 2.1.162, `langchain-core >= 1.0`
- Streaming requires `--verbose` with `--output-format stream-json`
- Event format: outer `stream_event` wraps inner `event`; unwrap then parse `content_block_delta`
- `usage_metadata` must include a `total_tokens` field

### Commit Convention

- One-line English, lowercase, no feat/fix prefix
- No `Co-Authored-By`
- See `.claude/COMMIT_CONVENTION.md`

### Adding a New Tool Function

In `claude_code_tool.py`:

```python
@tool
def my_new_tool(task: str, ...) -> str:
    """Tool description"""
    result = _run_claude_code(task, ...)
    return result.output
```

Follow the `claude_code` naming style (no extra prefixes) for consistency.

---

## License

MIT
