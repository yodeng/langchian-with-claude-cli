---
name: claude-code-cli
description: 通过 Claude Code CLI 子进程执行代码分析、文件操作、shell 命令等任务。当用户需要代码分析、项目结构探索、批量文件处理、多轮会话记忆任务时使用此 skill。
---

# Claude Code CLI Skill

将任务委托给独立的 Claude Code CLI 子进程，拥有文件系统、shell、git 完整访问能力。

## 模块位置

核心模块位于此 skill 目录下：
- `chat_claude_code.py` — `ChatClaudeCode`（LangChain BaseChatModel 适配器）
- `claude_code_tool.py` — `claude_code`（LangChain Tool 封装）

## 使用方式

### ChatClaudeCode（推荐）

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".claude/skills/claude-code-cli"))
from chat_claude_code import ChatClaudeCode

llm = ChatClaudeCode(working_dir="<目标目录>", effort="medium", timeout=300)

# 单轮调用
result = llm.invoke("<任务描述>")

# 多轮会话（同一实例自动保持上下文）
llm.invoke("列出所有 Python 文件")
llm.invoke("分析第一个文件的功能")

# 流式输出
for chunk in llm.stream("<任务描述>"):
    print(chunk.content, end="", flush=True)
```

### Delegator Tool

```python
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".claude/skills/claude-code-cli"))
from claude_code_tool import claude_code

result = claude_code.invoke({
    "task": "<任务描述>",
    "session_id": "my-session",  # 同一 session_id 保持上下文
    "effort": "high",
})
```

## 常用配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `working_dir` | 工作目录 | `.` |
| `effort` | low/medium/high/xhigh/max | medium |
| `timeout` | 超时秒数 | 300 |
| `allowed_tools` | 允许的工具列表 | 无限制 |
| `skip_permissions` | 跳过权限确认 | True |

## 典型场景

**代码分析**: `llm.invoke("分析项目架构，列出模块依赖关系")` — 用 `effort="high"`

**多轮记忆**: 同一 `ChatClaudeCode` 实例多次 `invoke`，自动通过 `--session-id` 保持上下文

**结构化输出**: `llm.with_structured_output(schema).invoke("...")` — 返回 JSON

**工具绑定**: `llm.bind_tools([my_tool]).invoke("...")` — Claude Code 原生执行工具

## 执行流程

1. 确定用户想要完成的具体任务
2. 选择合适的模式（ChatClaudeCode / Delegator）和参数
3. 编写并执行 Python 代码
4. 将结果以可读形式展示给用户
