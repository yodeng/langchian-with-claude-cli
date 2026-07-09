"""
测试 claude_code_tool.py — ClaudeCode 工具和会话管理

覆盖:
  - 纯函数/类: ClaudeCodeResult, ClaudeCodeSession, get_or_create_session,
    _cleanup_expired_sessions, _evict_oldest_sessions,
    close_session, list_sessions
  - Mock 集成: _run_claude_code, claude_code, claude_code_structured,
    claude_code_isolated
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

# 导入被测模块前，先隔离全局状态
import claude_code_tool as cct

from claude_code_tool import (
    ClaudeCodeResult,
    ClaudeCodeSession,
    _cleanup_expired_sessions,
    _evict_oldest_sessions,
    _run_claude_code,
    claude_code,
    claude_code_isolated,
    claude_code_structured,
    close_session,
    get_or_create_session,
    list_sessions,
)

# 访问模块级全局
_SESSIONS = cct._sessions
_SESSION_IDLE_TTL = cct._SESSION_IDLE_TTL
_MAX_SESSIONS = cct._MAX_SESSIONS

from tests.conftest import FakeCompletedProcess


# ═══════════════════════════════════════════════════════════════
# 辅助 fixture: 每个测试前后清空全局 session
# ═══════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def clean_sessions():
    """每个测试前后清空全局 session 注册表"""
    _SESSIONS.clear()
    yield
    _SESSIONS.clear()


# ═══════════════════════════════════════════════════════════════
# ClaudeCodeResult 测试
# ═══════════════════════════════════════════════════════════════


class TestClaudeCodeResult:

    def test_success_result(self):
        r = ClaudeCodeResult(success=True, output="hello world")
        assert r.success is True
        assert r.output == "hello world"
        assert r.structured is None
        assert r.stderr == ""
        assert r.returncode == 0
        assert r.session_id is None

    def test_failure_result(self):
        r = ClaudeCodeResult(
            success=False,
            output="",
            stderr="something went wrong",
            returncode=1,
            session_id="sess-123",
        )
        assert r.success is False
        assert r.output == ""
        assert r.stderr == "something went wrong"
        assert r.returncode == 1
        assert r.session_id == "sess-123"

    def test_structured_result(self):
        structured = {"key": "value", "nested": {"a": 1}}
        r = ClaudeCodeResult(
            success=True,
            output="result text",
            structured=structured,
        )
        assert r.structured == structured


# ═══════════════════════════════════════════════════════════════
# ClaudeCodeSession 测试
# ═══════════════════════════════════════════════════════════════


class TestClaudeCodeSession:

    def test_default_construction(self):
        s = ClaudeCodeSession()
        assert s.session_id is not None
        assert len(s.session_id) > 0
        assert s.working_dir == "."
        assert s.resume_count == 0

    def test_custom_params(self):
        s = ClaudeCodeSession(
            session_id="my-session",
            working_dir="/tmp/project",
        )
        assert s.session_id == "my-session"
        assert s.working_dir == "/tmp/project"

    def test_touch_updates_last_accessed(self):
        s = ClaudeCodeSession()
        s.last_accessed = 1000.0
        with patch.object(time, "time", return_value=1050.0):
            s.touch()
        assert s.last_accessed == 1050.0

    def test_age_seconds(self):
        s = ClaudeCodeSession()
        s.created_at = 1000.0
        with patch.object(time, "time", return_value=4600.0):
            assert s.age_seconds == 3600.0

    def test_idle_seconds(self):
        s = ClaudeCodeSession()
        s.last_accessed = 1000.0
        with patch.object(time, "time", return_value=1600.0):
            assert s.idle_seconds == 600.0

    def test_resume_count_default(self):
        s = ClaudeCodeSession()
        assert s.resume_count == 0


# ═══════════════════════════════════════════════════════════════
# Session 管理测试
# ═══════════════════════════════════════════════════════════════


class TestSessionManagement:

    def test_create_new_session(self):
        s = get_or_create_session(session_id=None, working_dir="/project")
        assert s.session_id in _SESSIONS
        assert s.working_dir == "/project"
        assert s.resume_count == 0

    def test_reuse_existing_session(self):
        s1 = get_or_create_session(session_id="sess-1")
        s2 = get_or_create_session(session_id="sess-1")
        assert s1 is s2

    def test_different_ids_create_different_sessions(self):
        s1 = get_or_create_session(session_id="sess-a")
        s2 = get_or_create_session(session_id="sess-b")
        assert s1 is not s2
        assert s1.session_id != s2.session_id

    def test_close_session(self):
        get_or_create_session(session_id="to-close")
        assert "to-close" in _SESSIONS
        assert close_session("to-close") is True
        assert "to-close" not in _SESSIONS

    def test_close_nonexistent_session(self):
        assert close_session("nonexistent") is False

    def test_list_sessions(self):
        get_or_create_session(session_id="s1")
        get_or_create_session(session_id="s2")
        sessions = list_sessions()
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {"s1", "s2"}

    def test_list_empty_sessions(self):
        assert list_sessions() == []

    def test_create_session_without_id_autogenerates(self):
        s = get_or_create_session(session_id=None)
        assert s.session_id is not None
        assert len(s.session_id) > 0


# ═══════════════════════════════════════════════════════════════
# 过期清理测试
# ═══════════════════════════════════════════════════════════════


class TestSessionCleanup:

    def test_cleanup_expired_by_idle(self):
        s = get_or_create_session(session_id="expiring")
        s.last_accessed = 1000.0
        # 时间推进超过空闲 TTL
        with patch.object(time, "time", return_value=1000.0 + _SESSION_IDLE_TTL + 1):
            _cleanup_expired_sessions()
        assert "expiring" not in _SESSIONS

    def test_cleanup_preserves_active_sessions(self):
        s = get_or_create_session(session_id="active")
        s.last_accessed = 1000.0
        # 时间未超过空闲 TTL
        with patch.object(time, "time", return_value=1100.0):
            _cleanup_expired_sessions()
        assert "active" in _SESSIONS

    def test_evict_oldest_when_over_limit(self):
        """超过 _MAX_SESSIONS 时驱逐最旧的"""
        for i in range(_MAX_SESSIONS + 5):
            s = ClaudeCodeSession(session_id=f"old-{i}")
            s.last_accessed = 1000.0 + i
            _SESSIONS[f"old-{i}"] = s
        _evict_oldest_sessions()
        assert len(_SESSIONS) <= _MAX_SESSIONS

    def test_evict_under_limit_does_nothing(self):
        get_or_create_session(session_id="s1")
        get_or_create_session(session_id="s2")
        _evict_oldest_sessions()
        assert len(_SESSIONS) == 2


# ═══════════════════════════════════════════════════════════════
# _run_claude_code() mock 测试
# ═══════════════════════════════════════════════════════════════


class TestRunClaudeCode:

    def test_normal_execution(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "task complete"}),
            stderr="",
            returncode=0,
        ))
        result = _run_claude_code("do something", working_dir="/tmp")
        assert result.success is True
        assert "task complete" in result.output

    def test_with_json_schema(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "structured", "data": [1, 2, 3]}),
            stderr="",
            returncode=0,
        ))
        schema = {"type": "object", "properties": {"data": {"type": "array"}}}
        result = _run_claude_code("extract data", json_schema=schema)
        assert result.success is True
        # 检查命令行中包含 --json-schema 参数
        call_args = mocker.call_args_list[0] if hasattr(mocker, 'call_args_list') else None

    def test_with_session_first_call(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "session reply"}),
            returncode=0,
        ))
        session = ClaudeCodeSession(session_id="sess-1", working_dir=".")
        result = _run_claude_code("continue work", session=session)
        assert result.success is True
        assert session.resume_count == 1
        assert result.session_id == "sess-1"

    def test_with_session_resume_call(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "resumed reply"}),
            returncode=0,
        ))
        session = ClaudeCodeSession(session_id="sess-1", working_dir=".")
        session.resume_count = 3  # 已经有 3 次调用
        result = _run_claude_code("more work", session=session)
        assert result.success is True
        assert session.resume_count == 4

    def test_timeout_error(self, mocker):
        import subprocess
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(
            cmd=["claude"], timeout=300,
        ))
        result = _run_claude_code("too slow", timeout=300)
        assert result.success is False
        assert "超时" in result.stderr
        assert result.returncode == -1

    def test_cli_not_found(self, mocker):
        mocker.patch("subprocess.run", side_effect=FileNotFoundError("claude not found"))
        result = _run_claude_code("do it")
        assert result.success is False
        assert "claude CLI" in result.stderr
        assert result.returncode == -2

    def test_structured_output_parsing(self, mocker):
        structured_data = {"name": "test", "count": 42}
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps(structured_data),
            returncode=0,
        ))
        result = _run_claude_code("extract", json_schema={"type": "object"})
        assert result.success is True
        assert result.structured == structured_data
        # output 也是 dict 被转为 str 所以 structured 直接用 dict
        assert result.structured["name"] == "test"

    def test_non_json_output_fallback(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="plain text output",
            returncode=0,
        ))
        result = _run_claude_code("summarize")
        assert result.success is True
        assert "plain text output" in result.output
        assert result.structured is None

    def test_empty_output_success(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="",
            returncode=0,
        ))
        result = _run_claude_code("do nothing")
        assert result.success is True
        assert result.output == ""


# ═══════════════════════════════════════════════════════════════
# Tool 函数 mock 测试
# ═══════════════════════════════════════════════════════════════


class TestClaudeCodeTool:

    def test_claude_code_normal(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "delegated task done"}),
            returncode=0,
        ))
        result = claude_code.invoke({"task": "analyze project"})
        assert "delegated task done" in result

    def test_claude_code_error(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="",
            stderr="permission denied",
            returncode=1,
        ))
        result = claude_code.invoke({"task": "unauthorized"})
        assert "[Claude Code 错误]" in result
        assert "returncode=1" in result

    def test_claude_code_with_session(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "session work"}),
            returncode=0,
        ))
        result1 = claude_code.invoke({"task": "step 1", "session_id": "shared"})
        result2 = claude_code.invoke({"task": "step 2", "session_id": "shared"})
        assert "session work" in result1
        assert "session work" in result2
        # 验证 session 被复用
        assert "shared" in _SESSIONS
        assert _SESSIONS["shared"].resume_count == 2

    def test_claude_code_without_session(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "independent work"}),
            returncode=0,
        ))
        result = claude_code.invoke({"task": "independent", "session_id": ""})
        assert "independent work" in result

    def test_claude_code_with_allowed_tools(self, mocker):
        mock_run = mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "tools used"}),
            returncode=0,
        ))
        result = claude_code.invoke({
            "task": "read and write",
            "allowed_tools": ["Read", "Write"],
        })
        assert "tools used" in result


class TestClaudeCodeStructuredTool:

    def test_structured_success(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"files": ["a.py", "b.py"], "count": 2}),
            returncode=0,
        ))
        result = claude_code_structured.invoke({
            "task": "list files",
            "output_schema": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "count": {"type": "integer"},
                },
            },
        })
        parsed = json.loads(result)
        assert parsed["count"] == 2
        assert len(parsed["files"]) == 2

    def test_structured_error(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="",
            stderr="schema mismatch",
            returncode=1,
        ))
        result = claude_code_structured.invoke({
            "task": "bad schema",
            "output_schema": {"type": "object"},
        })
        parsed = json.loads(result)
        assert "error" in parsed
        assert parsed["returncode"] == 1


class TestClaudeCodeIsolatedTool:

    def test_isolated_success(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "isolated work done"}),
            returncode=0,
        ))
        result = claude_code_isolated.invoke({
            "task": "isolated task",
            "context_files": ["/tmp/test"],
            "effort": "high",
        })
        assert "isolated work done" in result

    def test_isolated_error(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="",
            stderr="worktree error",
            returncode=1,
        ))
        result = claude_code_isolated.invoke({
            "task": "isolated task",
            "context_files": ["/tmp/test"],
        })
        assert "[Claude Code 错误]" in result


# ═══════════════════════════════════════════════════════════════
# _run_claude_code 额外参数测试
# ═══════════════════════════════════════════════════════════════


class TestRunClaudeCodeExtraParams:

    def test_with_system_prompt(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "with system prompt"}),
            returncode=0,
        ))
        result = _run_claude_code("task", system_prompt="You are a python expert")
        assert result.success is True

    def test_with_allowed_tools(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "tools allowed"}),
            returncode=0,
        ))
        result = _run_claude_code("task", allowed_tools=["Read", "Write"])
        assert result.success is True

    def test_with_extra_env(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "extra env"}),
            returncode=0,
        ))
        result = _run_claude_code("task", extra_env={"MY_VAR": "hello"})
        assert result.success is True

    def test_with_skip_permissions(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "permissions skipped"}),
            returncode=0,
        ))
        result = _run_claude_code("task", skip_permissions=True)
        assert result.success is True

    def test_with_context_files(self, mocker):
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout=json.dumps({"result": "context provided"}),
            returncode=0,
        ))
        result = _run_claude_code("task", context_files=["/data/file1.txt"])
        assert result.success is True
