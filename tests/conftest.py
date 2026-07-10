"""共享 fixtures — 供 test_chat_claude_code.py 和 test_claude_code_tool.py 使用"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


# ═══════════════════════════════════════════════════════════════
# 示例消息 fixtures
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def sample_human_message() -> HumanMessage:
    return HumanMessage(content="你好，帮我分析这个项目")


@pytest.fixture
def sample_system_message() -> SystemMessage:
    return SystemMessage(content="你是 Python 专家")


@pytest.fixture
def sample_ai_message() -> AIMessage:
    return AIMessage(content="好的，我来分析这个项目")


@pytest.fixture
def sample_ai_message_with_tool_calls() -> AIMessage:
    return AIMessage(
        content="让我读取文件",
        tool_calls=[
            {"name": "read_file", "args": {"path": "test.py"}, "id": "tc1"},
        ],
    )


@pytest.fixture
def sample_tool_message() -> ToolMessage:
    return ToolMessage(content="文件内容: print('hello')", tool_call_id="tc1", name="read_file")


@pytest.fixture
def sample_messages_mixed() -> list:
    """SystemMessage + HumanMessage + AIMessage 完整对话"""
    return [
        SystemMessage(content="你是助手"),
        HumanMessage(content="问题 1"),
        AIMessage(content="回答 1"),
        HumanMessage(content="问题 2"),
    ]


# ═══════════════════════════════════════════════════════════════
# Claude Code CLI 模拟输出
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def mock_claude_json_output() -> str:
    """模拟 claude --output-format json 的成功输出"""
    return json.dumps({
        "result": "这是 Claude Code 的回答",
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        },
    })


@pytest.fixture
def mock_claude_json_output_no_usage() -> str:
    return json.dumps({
        "result": "没有 usage 信息的回答",
    })


@pytest.fixture
def mock_claude_stream_output() -> str:
    """模拟 stream-json 输出的多行内容"""
    lines = [
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " World"},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }),
        json.dumps({
            "type": "stream_event",
            "event": {"type": "message_stop"},
        }),
    ]
    return "\n".join(lines)


@pytest.fixture
def mock_claude_error_output() -> str:
    return json.dumps({"result": "", "error": "something went wrong"})


# ═══════════════════════════════════════════════════════════════
# Mock subprocess fixtures
# ═══════════════════════════════════════════════════════════════


class FakeCompletedProcess:
    """模拟 subprocess.CompletedProcess"""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakePopen:
    """模拟 subprocess.Popen，提供 stdout 可迭代行"""

    def __init__(self, stdout_lines: list[str], stderr_lines: list[str] | None = None):
        self._stdout_lines = stdout_lines
        self._stderr_lines = stderr_lines or []
        self.stdout = _FakeLineIterable(self._stdout_lines)
        self.stderr = _FakeLineIterable(self._stderr_lines)
        self.terminated = False
        self.killed = False
        self._returncode: int | None = None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def poll(self) -> int | None:
        return self._returncode


class _FakeLineIterable:
    """模拟 stdout/stderr 的可迭代行读取"""

    def __init__(self, lines: list[str]):
        self._lines = lines
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._lines):
            raise StopIteration
        line = self._lines[self._index]
        self._index += 1
        return line


class FakeAsyncProcess:
    """模拟 asyncio.subprocess.Process"""

    def __init__(
        self,
        stdout_data: bytes = b"",
        stderr_data: bytes = b"",
        returncode: int = 0,
        stdout_lines: list[bytes] | None = None,
    ):
        self._stdout_data = stdout_data
        self._stderr_data = stderr_data
        self.returncode = returncode
        self._stdout_lines = stdout_lines or []
        self.terminated = False
        self.killed = False
        # 预先创建 reader，避免每次访问 stdout 创建新实例
        self.stdout = _FakeAsyncStreamReader(self._stdout_lines)

    async def communicate(self):
        return self._stdout_data, self._stderr_data

    async def wait(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class _FakeAsyncStreamReader:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self._index = 0

    async def readline(self) -> bytes:
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line


@pytest.fixture
def mock_subprocess_run(mocker):
    """Mock subprocess.run 返回成功结果"""
    mock = mocker.patch("subprocess.run")
    mock.return_value = FakeCompletedProcess(
        stdout=json.dumps({"result": "mocked response"}),
        stderr="",
        returncode=0,
    )
    return mock


@pytest.fixture
def mock_subprocess_popen(mocker, mock_claude_stream_output):
    """Mock subprocess.Popen 返回流式结果"""
    mock = mocker.patch("subprocess.Popen")
    lines = mock_claude_stream_output.split("\n")
    mock.return_value = FakePopen(stdout_lines=lines)
    return mock


@pytest.fixture
def mock_asyncio_subprocess(mocker):
    """Mock asyncio.create_subprocess_exec 返回成功结果"""
    mock = mocker.patch("asyncio.create_subprocess_exec")
    stdout = json.dumps({"result": "async mocked response"}).encode("utf-8")
    mock.return_value = FakeAsyncProcess(
        stdout_data=stdout,
        stderr_data=b"",
        returncode=0,
    )
    return mock
