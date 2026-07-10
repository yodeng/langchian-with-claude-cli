# CLAUDE.md

## 项目概述

`langchain-with-claude-cli` — 将 Claude Code CLI 包装为 LangChain 生态组件，提供三种后端集成方式。

## 项目结构

```
chat_claude_code.py      # 后端 1: ChatClaudeCode — BaseChatModel 子类
claude_code_tool.py      # 后端 2: claude_code 系列 — LangChain @tool 封装
pyproject.toml           # 项目配置，依赖声明
README.md                # 英文 README（默认）
README_zh.md             # 中文 README
agents/                  # LangGraph Agent 实现
  code_analysis_agent.py # 三阶段代码分析流水线
  demo_memory_agent.py   # 多轮对话记忆演示
examples/                # 使用示例
  example_chat_model.py  # ChatClaudeCode 7 种使用模式
  example_tool.py        # Tool 委托模式 4 种用法
tests/                   # 测试（141 个用例）
  test_chat_claude_code.py
  test_claude_code_tool.py
.claude/skills/          # 后端 3: Claude Code 自身 skill 系统
  claude-code-cli/       # Claude Code CLI 调用 skill
```

## 三种后端

### 后端 1: ChatModel — `ChatClaudeCode` (chat_claude_code.py)

继承 `BaseChatModel`，与 `ChatOpenAI` 接口兼容，适合做 Agent 的 LLM 节点。

- 底层通过 `claude -p` 子进程调用，拥有完整文件系统、shell、git 能力
- `_generate` / `_stream` / `_agenerate` / `_astream` 全部实现
- 同一实例内自动通过 `--session-id` / `--resume` 保持多轮上下文
- `bind_tools()` 注入工具描述到 system prompt，Claude Code 原生执行工具
- `with_structured_output()` 通过 `--json-schema` 约束输出
- `reset_session()` 清空记忆
- 默认 timeout: 600s，范围 10-3600s

### 后端 2: Tool — `claude_code` 系列 (claude_code_tool.py)

`@tool` 装饰的独立工具函数，适合让 LangGraph Agent 在特定步骤上委托给 Claude Code。

- `claude_code` — 基础委托，返回文本
- `claude_code_structured` — 结构化 JSON 输出（`--json-schema`）
- `claude_code_isolated` — git worktree 隔离执行
- `claude_code_streaming` — 流式输出（返回 `subprocess.Popen`）

### 后端 3: Skill — `claude-code-cli` (.claude/skills/)

Claude Code 自身的 skill，让 Claude Code 以子进程方式调用另一个 Claude Code 实例。

- 子进程隔离（独立 session、独立 worktree）
- 多级调用（外层 Claude Code 调度内层 Claude Code）
- 通过符号链接复用核心模块

## 技术要点

- 依赖 `claude` CLI >= 2.1.162，langchain-core >= 1.0，pydantic >= 2.0
- 流式输出需要 `--verbose` 配合 `--output-format stream-json`
- 事件格式：外层 `stream_event` 包装内层 `event`，需解包后解析 `content_block_delta`
- `usage_metadata` 必须包含 `total_tokens` 字段
- `stderr` 在流式模式下使用 `DEVNULL`，避免管道缓冲区阻塞
- 临时 schema 文件的写入放在 try 块内，确保异常时也能在 finally 中清理


## README 维护

- `README.md` — 英文版（默认）
- `README_zh.md` — 中文版
- 两份文件需同步更新，包含 `vs deep_agent` 对比章节
- 语言切换格式：`[English](./README.md) | [中文](./README_zh.md)`
