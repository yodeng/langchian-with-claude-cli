"""
Claude Code Delegator — 将 Claude Code CLI 包装为 LangChain Tool

让 LangChain / LangGraph Agent 在任意步骤上委托给 Claude Code 执行，
利用其强大的文件操作、shell 执行、代码生成能力。

核心能力：
  - 单次委托：claude -p "prompt" --output-format json
  - 会话延续：--session-id 保持多步上下文
  - 结构化输出：--json-schema 约束返回格式
  - 隔离执行：--worktree 创建独立 git 工作区
  - 权限控制：--dangerously-skip-permissions / --allowedTools
  - 文件传递：--add-dir 授权目录 + stdin 或临时文件传递上下文

依赖：claude CLI >= 2.1.162, langchain-core, langgraph
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════

@dataclass
class ClaudeCodeResult:
    """Claude Code 调用结果"""
    success: bool
    output: str
    structured: dict | None = None
    stderr: str = ""
    returncode: int = 0
    session_id: str | None = None


@dataclass
class ClaudeCodeSession:
    """管理一个 Claude Code 会话，支持多步延续"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    working_dir: str = "."
    resume_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """更新最后访问时间"""
        self.last_accessed = time.time()

    @property
    def age_seconds(self) -> float:
        """会话创建以来的秒数"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """上次访问以来的空闲秒数"""
        return time.time() - self.last_accessed


# ═══════════════════════════════════════════
# 核心执行函数
# ═══════════════════════════════════════════

def _run_claude_code(
    prompt: str,
    *,
    session: ClaudeCodeSession | None = None,
    working_dir: str = ".",
    allowed_tools: list[str] | None = None,
    json_schema: dict | None = None,
    system_prompt: str | None = None,
    effort: str = "medium",
    timeout: int = 300,
    use_worktree: bool = False,
    skip_permissions: bool = False,
    context_files: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> ClaudeCodeResult:
    """
    执行一次 Claude Code 调用。

    Args:
        prompt: 任务描述
        session: 会话对象，传入则延续之前的会话
        working_dir: 工作目录
        allowed_tools: 允许的工具列表，如 ["Read", "Write", "Bash(git *)"]
        json_schema: JSON Schema，约束 Claude Code 的输出格式
        system_prompt: 自定义 system prompt
        effort: 努力级别 (low/medium/high/xhigh/max)
        timeout: 超时秒数
        use_worktree: 是否创建 git worktree 隔离
        skip_permissions: 跳过权限确认（自动化场景）
        context_files: 需要授权访问的额外文件/目录
        extra_env: 额外的环境变量

    Returns:
        ClaudeCodeResult
    """
    cmd = ["claude", "-p", prompt]
    working = working_dir

    # === 输出格式 ===
    schema_file = None
    if json_schema:
        cmd.extend(["--output-format", "json"])
        schema_file = Path(working) / f".claude_schema_{str(uuid.uuid4())}.json"
        schema_file.write_text(json.dumps(json_schema), encoding="utf-8")
        cmd.extend(["--json-schema", str(schema_file)])
    else:
        cmd.extend(["--output-format", "json"])

    # === 会话管理 ===
    if session is not None:
        if session.resume_count == 0:
            cmd.extend(["--session-id", session.session_id])
        else:
            cmd.extend(["--resume", session.session_id])
        session.resume_count += 1

    # === 隔离模式 ===
    if use_worktree:
        wt_name = f"lc-delegate-{uuid.uuid4().hex[:8]}"
        cmd.extend(["--worktree", wt_name])
        # worktree 创建后会切换 working_dir，但我们的 cwd 不变
        # 实际执行时 worktree 路径由 Claude Code 管理

    # === 工具限制 ===
    if allowed_tools:
        cmd.append("--disallowedTools")
        cmd.append("Bash(curl *)")  # 默认禁用网络
        cmd.append("--allowedTools")
        cmd.extend(allowed_tools)

    # === 权限 ===
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")

    # === 上下文文件 ===
    if context_files:
        cmd.append("--add-dir")
        cmd.extend(context_files)

    # === 努力级别 ===
    cmd.extend(["--effort", effort])

    # === 系统提示 ===
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    # === 环境变量 ===
    env = os.environ.copy()
    env["CLAUDE_CODE_SIMPLE"] = "1"
    if extra_env:
        env.update(extra_env)

    # === 执行 ===
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working,
            env=env,
        )

        output = result.stdout
        structured = None
        error = result.stderr

        # 尝试解析 JSON 输出
        if output.strip():
            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict):
                    structured = parsed
                    # 提取 readable 文本
                    output = parsed.get("result", parsed.get("content", output))
                    if isinstance(output, list):
                        output = "\n".join(
                            block.get("text", str(block))
                            for block in output
                            if isinstance(block, dict)
                        )
            except json.JSONDecodeError:
                logger.debug("JSON 解析失败，使用原始文本输出")

        return ClaudeCodeResult(
            success=result.returncode == 0,
            output=output.strip() if output else "",
            structured=structured,
            stderr=error,
            returncode=result.returncode,
            session_id=session.session_id if session else None,
        )

    except subprocess.TimeoutExpired:
        return ClaudeCodeResult(
            success=False,
            output="",
            stderr=f"超时（{timeout}秒）",
            returncode=-1,
        )
    except FileNotFoundError:
        return ClaudeCodeResult(
            success=False,
            output="",
            stderr="claude CLI 未找到，请确认已安装 Claude Code",
            returncode=-2,
        )
    finally:
        # 清理临时 schema 文件
        if schema_file and schema_file.exists():
            try:
                schema_file.unlink()
            except OSError:
                pass


# ═══════════════════════════════════════════
# LangChain Tool 定义
# ═══════════════════════════════════════════

# 全局 session 注册表（按 session_id 索引）
_sessions: dict[str, ClaudeCodeSession] = {}

# session 生命周期配置
_MAX_SESSIONS: int = 100
_SESSION_IDLE_TTL: float = 3600.0  # 空闲 1 小时后过期
_SESSION_MAX_AGE: float = 86400.0  # 最长存活 24 小时


def _cleanup_expired_sessions() -> None:
    """清理过期和空闲的 session"""
    now = time.time()
    expired = [
        sid
        for sid, s in _sessions.items()
        if now - s.last_accessed > _SESSION_IDLE_TTL
        or now - s.created_at > _SESSION_MAX_AGE
    ]
    for sid in expired:
        del _sessions[sid]
        logger.debug("清理过期 session: %s", sid)


def _evict_oldest_sessions() -> None:
    """超过最大数量时驱逐最旧的 session"""
    if len(_sessions) <= _MAX_SESSIONS:
        return
    excess = len(_sessions) - _MAX_SESSIONS
    oldest = sorted(_sessions.items(), key=lambda kv: kv[1].last_accessed)[:excess]
    for sid, _ in oldest:
        del _sessions[sid]
        logger.debug("驱逐旧 session: %s", sid)


def get_or_create_session(
    session_id: str | None = None,
    working_dir: str = ".",
) -> ClaudeCodeSession:
    """获取或创建 Claude Code 会话，自动清理过期和超量 session"""
    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        session.touch()
        return session

    _cleanup_expired_sessions()
    _evict_oldest_sessions()

    session = ClaudeCodeSession(
        session_id=session_id or str(uuid.uuid4()),
        working_dir=working_dir,
    )
    _sessions[session.session_id] = session
    return session


@tool
def delegate_to_claude_code(
    task: str,
    session_id: str = "",
    allowed_tools: list[str] | None = None,
    effort: str = "medium",
) -> str:
    """
    将任务委托给 Claude Code CLI 执行。适合需要以下能力的步骤：
    - 读写文件、搜索代码库
    - 执行 git / shell 命令
    - 复杂的代码分析、重构、生成
    - 项目级理解（读取多个文件并综合）

    Args:
        task: 详细的任务描述。越具体越好，应包括期望的输出格式。
        session_id: 会话 ID。传空字符串则每次独立调用；
                    传入相同 ID 则在同一会话内延续（有上下文记忆）。
        allowed_tools: 允许的工具列表。空列表 = 使用默认限制。
                       例如: ["Read", "Write", "Bash(git diff)", "Bash(git log)"]
        effort: 努力级别。low=快速 low=快速 / medium=平衡 / high=深入

    Returns:
        Claude Code 的执行结果（文本）
    """
    session = get_or_create_session(session_id) if session_id else None

    result = _run_claude_code(
        task,
        session=session,
        working_dir=".",
        allowed_tools=allowed_tools if allowed_tools else None,
        effort=effort,
        skip_permissions=True,  # Agent 调用场景下跳过权限确认
    )

    if not result.success:
        return f"[Claude Code 错误] returncode={result.returncode}\n{result.stderr}"

    return result.output


@tool
def delegate_to_claude_code_structured(
    task: str,
    output_schema: dict,
    session_id: str = "",
    effort: str = "medium",
) -> str:
    """
    将任务委托给 Claude Code 并要求结构化 JSON 输出。
    适合需要从代码库中提取结构化信息的步骤。

    Args:
        task: 详细的任务描述。Claude Code 会按 output_schema 约束的格式返回。
        output_schema: JSON Schema 定义期望的输出结构。
                       例如: {"type": "object", "properties": {"files": {"type": "array", ...}}}
        session_id: 会话 ID，同 delegate_to_claude_code
        effort: 努力级别

    Returns:
        JSON 字符串（符合 output_schema 约束）
    """
    session = get_or_create_session(session_id) if session_id else None

    result = _run_claude_code(
        task,
        session=session,
        working_dir=".",
        json_schema=output_schema,
        effort=effort,
        skip_permissions=True,
    )

    if not result.success:
        return json.dumps({"error": result.stderr, "returncode": result.returncode})

    if result.structured:
        return json.dumps(result.structured, ensure_ascii=False, indent=2)

    return result.output


@tool
def delegate_to_claude_code_isolated(
    task: str,
    context_files: list[str],
    effort: str = "high",
) -> str:
    """
    在隔离的 git worktree 中将任务委托给 Claude Code 执行。
    适合并行执行多个互不干扰的任务（如同时分析多个模块）。

    Args:
        task: 详细的任务描述
        context_files: 需要授权访问的文件/目录路径列表
        effort: 努力级别

    Returns:
        Claude Code 的执行结果
    """
    result = _run_claude_code(
        task,
        working_dir=".",
        use_worktree=True,
        context_files=context_files,
        effort=effort,
        skip_permissions=True,
    )

    if not result.success:
        return f"[Claude Code 错误] returncode={result.returncode}\n{result.stderr}"

    return result.output


# ═══════════════════════════════════════════
# 高级：流式调用（用于实时反馈）
# ═══════════════════════════════════════════

def delegate_to_claude_code_streaming(
    prompt: str,
    session: ClaudeCodeSession | None = None,
    working_dir: str = ".",
    timeout: int = 300,
) -> "subprocess.Popen[str]":
    """
    启动 Claude Code 流式调用，返回 Popen 对象供逐行读取。

    用法:
        with delegate_to_claude_code_streaming("分析这个项目") as proc:
            for line in proc.stdout:
                print(line, end="")

    Args:
        prompt: 任务描述
        session: 会话对象
        working_dir: 工作目录
        timeout: 超时秒数

    Returns:
        subprocess.Popen 对象，stdout 为流式 JSON 行
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--include-partial-messages",
        "--verbose",  # stream-json 要求 --verbose
    ]

    if session:
        if session.resume_count == 0:
            cmd.extend(["--session-id", session.session_id])
        else:
            cmd.extend(["--resume", session.session_id])
        session.resume_count += 1

    env = os.environ.copy()

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=working_dir,
        env=env,
    )


# ═══════════════════════════════════════════
# 会话管理工具
# ═══════════════════════════════════════════

def close_session(session_id: str) -> bool:
    """关闭并清理指定会话"""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> list[dict]:
    """列出所有活跃会话"""
    return [
        {"session_id": s.session_id, "working_dir": s.working_dir, "resume_count": s.resume_count}
        for s in _sessions.values()
    ]
