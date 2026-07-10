"""
测试 chat_claude_code.py — ChatClaudeCode 和相关纯函数

覆盖:
  - 纯函数: _format_messages, _parse_json_output, _extract_text_from_response,
    _extract_usage, _parse_json_line, _parse_stream_event_line,
    _check_stop_words, _stream_event_to_chunk, _build_chat_result,
    _build_command, _build_tool_description, _dict_to_tool
  - ChatClaudeCode 类: __init__, reset_session, bind_tools,
    with_structured_output, _generate, _stream, _agenerate, _astream
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool as langchain_tool

from chat_claude_code import (
    ChatClaudeCode,
    _build_chat_result,
    _build_tool_description,
    _check_stop_words,
    _dict_to_tool,
    _extract_text_from_response,
    _extract_usage,
    _format_messages,
    _parse_json_line,
    _parse_json_output,
    _parse_stream_event_line,
    _stream_event_to_chunk,
)

# 导入 conftest 中的辅助类
from tests.conftest import FakeCompletedProcess, FakePopen, FakeAsyncProcess


# ═══════════════════════════════════════════════════════════════
# _format_messages() 测试
# ═══════════════════════════════════════════════════════════════


class TestFormatMessages:
    """消息格式化 — 纯函数，无需 mock"""

    def test_empty_list_returns_empty(self):
        conv, sys_prompt = _format_messages([])
        assert conv == ""
        assert sys_prompt is None

    def test_single_human_message(self, sample_human_message):
        conv, sys_prompt = _format_messages([sample_human_message])
        assert "Human: 你好，帮我分析这个项目" in conv
        assert sys_prompt is None

    def test_single_system_message(self, sample_system_message):
        conv, sys_prompt = _format_messages([sample_system_message])
        assert conv == ""
        assert "Python 专家" in sys_prompt

    def test_human_and_ai_mixed(self):
        messages = [
            HumanMessage(content="问题 1"),
            AIMessage(content="回答 1"),
            HumanMessage(content="问题 2"),
        ]
        conv, sys_prompt = _format_messages(messages)
        assert "Human: 问题 1" in conv
        assert "Assistant: 回答 1" in conv
        assert "Human: 问题 2" in conv
        assert sys_prompt is None

    def test_full_conversation_with_system(self, sample_messages_mixed):
        conv, sys_prompt = _format_messages(sample_messages_mixed)
        assert "你是助手" in sys_prompt
        assert "Human: 问题 1" in conv
        assert "Assistant: 回答 1" in conv
        assert "Human: 问题 2" in conv

    def test_ai_message_with_tool_calls(self, sample_ai_message_with_tool_calls):
        conv, sys_prompt = _format_messages([sample_ai_message_with_tool_calls])
        assert "Assistant:" in conv
        assert "调用工具: read_file" in conv
        assert '{"path": "test.py"}' in conv

    def test_tool_message_formatting(self, sample_tool_message):
        conv, sys_prompt = _format_messages([sample_tool_message])
        assert "Tool result (read_file)" in conv
        assert "print('hello')" in conv
        assert sys_prompt is None

    def test_multiple_system_messages_merged(self):
        messages = [
            SystemMessage(content="规则 1"),
            SystemMessage(content="规则 2"),
            HumanMessage(content="问题"),
        ]
        conv, sys_prompt = _format_messages(messages)
        assert "规则 1" in sys_prompt
        assert "规则 2" in sys_prompt
        assert "Human: 问题" in conv

    def test_human_message_with_name(self):
        msg = HumanMessage(content="你好", name="小明")
        conv, _ = _format_messages([msg])
        assert "Human (as 小明): 你好" in conv


# ═══════════════════════════════════════════════════════════════
# _parse_json_output() 测试
# ═══════════════════════════════════════════════════════════════


class TestParseJsonOutput:

    def test_valid_json_dict(self):
        result = _parse_json_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_fallback(self):
        result = _parse_json_output("这不是 JSON")
        assert result == {"result": "这不是 JSON"}

    def test_json_array_wrapped(self):
        """非 dict 的 JSON 被包装为 {"result": stdout}"""
        result = _parse_json_output('[1, 2, 3]')
        assert result == {"result": "[1, 2, 3]"}

    def test_empty_string(self):
        result = _parse_json_output("")
        assert result == {"result": ""}


# ═══════════════════════════════════════════════════════════════
# _extract_text_from_response() 测试
# ═══════════════════════════════════════════════════════════════


class TestExtractTextFromResponse:

    def test_result_as_string(self):
        assert _extract_text_from_response({"result": "hello"}) == "hello"

    def test_result_as_list_of_dicts(self):
        parsed = {
            "result": [
                {"type": "text", "text": "第一段"},
                {"type": "tool_use", "name": "read", "input": {"path": "f.py"}},
                {"type": "tool_result", "content": "ok"},
                {"type": "text", "text": "第二段"},
            ]
        }
        text = _extract_text_from_response(parsed)
        assert "第一段" in text
        assert "[Tool: read" in text
        assert "[Tool Result: ok]" in text
        assert "第二段" in text

    def test_result_as_list_of_strings(self):
        parsed = {"result": ["line1", "line2"]}
        assert _extract_text_from_response(parsed) == "line1\nline2"

    def test_fallback_to_int_result(self):
        assert _extract_text_from_response({"result": 42}) == "42"

    def test_no_result_key(self):
        assert _extract_text_from_response({"other": "data"}) == ""

    def test_result_key_none(self):
        assert _extract_text_from_response({"result": None}) == "None"


# ═══════════════════════════════════════════════════════════════
# _extract_usage() 测试
# ═══════════════════════════════════════════════════════════════


class TestExtractUsage:

    def test_with_usage(self):
        parsed = {"result": "ok", "usage": {"input_tokens": 10, "output_tokens": 5}}
        assert _extract_usage(parsed) == {"input_tokens": 10, "output_tokens": 5}

    def test_without_usage(self):
        assert _extract_usage({"result": "ok"}) is None

    def test_empty_dict(self):
        assert _extract_usage({}) is None


# ═══════════════════════════════════════════════════════════════
# _parse_json_line() 测试
# ═══════════════════════════════════════════════════════════════


class TestParseJsonLine:

    def test_valid_json(self):
        assert _parse_json_line('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_none(self):
        assert _parse_json_line("not json") is None

    def test_empty_string(self):
        assert _parse_json_line("") is None


# ═══════════════════════════════════════════════════════════════
# _parse_stream_event_line() 测试
# ═══════════════════════════════════════════════════════════════


class TestParseStreamEventLine:

    def test_system_event_returns_none(self):
        line = json.dumps({"type": "system", "message": "init"})
        assert _parse_stream_event_line(line) is None

    def test_stream_event_text_delta(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        })
        result = _parse_stream_event_line(line)
        assert result == {"type": "text", "content": "Hello"}

    def test_stream_event_input_json_delta(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"key":'},
            },
        })
        result = _parse_stream_event_line(line)
        assert result == {"type": "text", "content": '{"key":'}

    def test_message_delta_usage(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {
                "type": "message_delta",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            },
        })
        result = _parse_stream_event_line(line)
        assert result == {
            "type": "usage",
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }

    def test_message_stop(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {"type": "message_stop"},
        })
        result = _parse_stream_event_line(line)
        assert result == {"type": "stop"}

    def test_invalid_json_line(self):
        assert _parse_stream_event_line("not json") is None

    def test_direct_event_without_stream_wrapper(self):
        """event 直接在顶层，没有 stream_event 包装"""
        line = json.dumps({
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "direct"},
        })
        result = _parse_stream_event_line(line)
        assert result == {"type": "text", "content": "direct"}

    def test_empty_text_delta_returns_none(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": ""},
            },
        })
        assert _parse_stream_event_line(line) is None

    def test_empty_input_json_delta_returns_none(self):
        line = json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": ""},
            },
        })
        assert _parse_stream_event_line(line) is None


# ═══════════════════════════════════════════════════════════════
# _check_stop_words() 测试
# ═══════════════════════════════════════════════════════════════


class TestCheckStopWords:

    def test_stop_word_found(self):
        assert _check_stop_words(["hello", " world"], ["world"]) is True

    def test_stop_word_not_found(self):
        assert _check_stop_words(["hello", " world"], ["goodbye"]) is False

    def test_empty_collected_text(self):
        assert _check_stop_words([], ["stop"]) is False

    def test_partial_match_in_collected(self):
        """collected 是分片列表，stop 词横跨两个片"""
        assert _check_stop_words(["hel", "lo"], ["hello"]) is True


# ═══════════════════════════════════════════════════════════════
# _stream_event_to_chunk() 测试
# ═══════════════════════════════════════════════════════════════


class TestStreamEventToChunk:

    def test_text_event(self):
        chunk = _stream_event_to_chunk({"type": "text", "content": "hello"})
        assert isinstance(chunk, ChatGenerationChunk)
        assert chunk.message.content == "hello"

    def test_usage_event(self):
        chunk = _stream_event_to_chunk({
            "type": "usage",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        })
        assert isinstance(chunk, ChatGenerationChunk)
        assert chunk.message.content == ""
        assert chunk.message.usage_metadata == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }

    def test_unknown_event_returns_none(self):
        assert _stream_event_to_chunk({"type": "unknown"}) is None


# ═══════════════════════════════════════════════════════════════
# _build_chat_result() 测试
# ═══════════════════════════════════════════════════════════════


class TestBuildChatResult:

    def test_success_result(self, mock_claude_json_output):
        result = _build_chat_result(mock_claude_json_output, "", 0)
        assert result is not None
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        gen = result.generations[0]
        assert isinstance(gen, ChatGeneration)
        assert "Claude Code" in gen.message.content
        # 检查 llm_output 含 usage
        assert result.llm_output is not None
        assert "usage" in result.llm_output

    def test_error_result(self):
        result = _build_chat_result(
            "", "something went wrong", 1
        )
        assert result is not None
        assert "[Claude Code 错误]" in result.generations[0].message.content
        assert result.llm_output is not None
        assert result.llm_output.get("returncode") == 1

    def test_result_without_usage(self, mock_claude_json_output_no_usage):
        result = _build_chat_result(mock_claude_json_output_no_usage, "", 0)
        assert result is not None
        assert result.llm_output == {}


# ═══════════════════════════════════════════════════════════════
# _build_command() 测试
# ═══════════════════════════════════════════════════════════════


class TestBuildCommand:

    @pytest.fixture
    def llm(self):
        """创建一个基础 ChatClaudeCode 实例用于测试 _build_command"""
        return ChatClaudeCode(working_dir=".", effort="medium")

    def test_basic_command(self, llm):
        cmd = llm._build_command("hello")
        assert cmd[0] == "claude"
        assert cmd[1] == "-p"
        assert cmd[2] == "hello"
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--session-id" in cmd
        assert "--effort" in cmd
        assert "medium" in cmd

    def test_resume_session(self, llm):
        llm._session_turn = 1
        cmd = llm._build_command("continue")
        assert "--resume" in cmd
        assert "--session-id" not in cmd

    def test_stream_mode(self, llm):
        cmd = llm._build_command("stream this", stream=True)
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--include-partial-messages" in cmd
        assert "--verbose" in cmd

    def test_json_schema_mode(self, llm):
        llm._output_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        cmd = llm._build_command("structured")
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "--json-schema" in cmd
        schema_idx = cmd.index("--json-schema")
        assert schema_idx + 1 < len(cmd)
        schema_str = cmd[schema_idx + 1]
        parsed = json.loads(schema_str)
        assert parsed["type"] == "object"

    def test_allowed_tools(self, llm):
        llm.allowed_tools = ["Read", "Write"]
        cmd = llm._build_command("task")
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert "Read" in cmd[idx:]
        assert "Write" in cmd[idx:]

    def test_disallowed_tools(self, llm):
        llm.disallowed_tools = ["Bash(curl *)"]
        cmd = llm._build_command("task")
        assert "--disallowedTools" in cmd
        idx = cmd.index("--disallowedTools")
        assert "Bash(curl *)" in cmd[idx:]

    def test_model_and_max_tokens(self, llm):
        llm.model = "deepseek-v4-pro"
        llm.max_tokens = 4096
        cmd = llm._build_command("task")
        assert "--model" in cmd
        assert "deepseek-v4-pro" in cmd
        assert "--max-tokens" in cmd
        assert "4096" in cmd

    def test_skip_permissions(self, llm):
        llm.skip_permissions = True
        cmd = llm._build_command("task")
        assert "--dangerously-skip-permissions" in cmd

    def test_context_files(self, llm):
        llm.context_files = ["/tmp/test", "/data/project"]
        cmd = llm._build_command("task")
        assert "--add-dir" in cmd
        idx = cmd.index("--add-dir")
        assert "/tmp/test" in cmd[idx:]
        assert "/data/project" in cmd[idx:]

    def test_system_prompt_inline(self, llm):
        cmd = llm._build_command("task", system_prompt="自定义提示")
        assert "--system-prompt" in cmd
        assert "自定义提示" in cmd

    def test_system_prompt_from_instance(self, llm):
        llm.system_prompt = "实例系统提示"
        cmd = llm._build_command("task")
        assert "--system-prompt" in cmd
        assert "实例系统提示" in cmd

    def test_extra_env_not_in_cmd(self, llm):
        """extra_env 不影响 cmd 参数，只影响 env dict"""
        llm.extra_env = {"MY_VAR": "value"}
        cmd = llm._build_command("task")
        assert "MY_VAR" not in cmd


# ═══════════════════════════════════════════════════════════════
# _build_tool_description() 测试
# ═══════════════════════════════════════════════════════════════


class TestBuildToolDescription:

    def test_single_tool(self):
        @langchain_tool
        def my_tool(x: str) -> str:
            """我的工具描述"""
            return x

        desc = _build_tool_description([my_tool])
        assert "my_tool" in desc
        assert "我的工具描述" in desc

    def test_multiple_tools(self):
        @langchain_tool
        def tool_a(x: int) -> int:
            """工具 A"""
            return x

        @langchain_tool
        def tool_b(y: str) -> str:
            """工具 B"""
            return y

        desc = _build_tool_description([tool_a, tool_b])
        assert "tool_a" in desc
        assert "工具 A" in desc
        assert "tool_b" in desc
        assert "工具 B" in desc

    def test_empty_tools(self):
        desc = _build_tool_description([])
        assert desc == ""


# ═══════════════════════════════════════════════════════════════
# _dict_to_tool() 测试
# ═══════════════════════════════════════════════════════════════


class TestDictToTool:

    def test_basic_dict(self):
        tool = _dict_to_tool({
            "name": "my_func",
            "description": "does something",
            "func": lambda: "ok",
        })
        assert tool.name == "my_func"
        assert tool.description == "does something"

    def test_dict_with_args(self):
        def sample_func(x: int, y: str = "default") -> str:
            """sample desc"""
            return f"{x}-{y}"

        tool = _dict_to_tool({
            "name": "func_with_args",
            "description": "test func",
            "func": sample_func,
        })
        assert tool.name == "func_with_args"


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode 基础测试（无需 subprocess）
# ═══════════════════════════════════════════════════════════════


class TestChatClaudeCodeBasic:

    def test_init_generates_session_id(self):
        llm = ChatClaudeCode()
        assert llm._session_id is not None
        assert len(llm._session_id) > 0

    def test_init_session_turn_zero(self):
        llm = ChatClaudeCode()
        assert llm._session_turn == 0

    def test_llm_type(self):
        llm = ChatClaudeCode()
        assert llm._llm_type == "claude-code-cli"

    def test_identifying_params(self):
        llm = ChatClaudeCode(working_dir="/test", effort="high", model="deepseek-v4-pro")
        params = llm._identifying_params
        assert params["working_dir"] == "/test"
        assert params["effort"] == "high"
        assert params["model"] == "deepseek-v4-pro"

    def test_reset_session_changes_id(self):
        llm = ChatClaudeCode()
        old_id = llm._session_id
        new_id = llm.reset_session()
        assert new_id != old_id
        assert llm._session_turn == 0

    def test_reset_session_returns_string(self):
        llm = ChatClaudeCode()
        assert isinstance(llm.reset_session(), str)


# ═══════════════════════════════════════════════════════════════
# bind_tools() 测试
# ═══════════════════════════════════════════════════════════════


class TestBindTools:

    def test_bind_tools_adds_description_to_system_prompt(self):
        llm = ChatClaudeCode()

        @langchain_tool
        def search(query: str) -> str:
            """Search the web"""
            return query

        llm_with_tools = llm.bind_tools([search])
        assert llm_with_tools.system_prompt is not None
        assert "search" in llm_with_tools.system_prompt
        assert "可用工具" in llm_with_tools.system_prompt

    def test_bind_tools_merges_with_existing_prompt(self):
        llm = ChatClaudeCode(system_prompt="现有系统提示")

        @langchain_tool
        def calc(x: int) -> int:
            """Calculate stuff"""
            return x

        llm_with_tools = llm.bind_tools([calc])
        assert "现有系统提示" in llm_with_tools.system_prompt
        assert "calc" in llm_with_tools.system_prompt

    def test_bind_tools_preserves_original(self):
        """原始实例不受 bind_tools 影响"""
        llm = ChatClaudeCode()
        original_prompt = llm.system_prompt

        @langchain_tool
        def dummy_tool() -> str:
            """Does nothing"""
            return ""
        llm.bind_tools([dummy_tool])
        assert llm.system_prompt == original_prompt

    def test_bind_tools_generates_new_session(self):
        llm = ChatClaudeCode()
        old_id = llm._session_id

        @langchain_tool
        def dummy_tool() -> str:
            """Does nothing"""
            return ""
        new_llm = llm.bind_tools([dummy_tool])
        assert new_llm._session_id != old_id


# ═══════════════════════════════════════════════════════════════
# with_structured_output() 测试
# ═══════════════════════════════════════════════════════════════


class TestWithStructuredOutput:

    def test_dict_schema(self):
        llm = ChatClaudeCode()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        new_llm = llm.with_structured_output(schema)
        assert new_llm._output_schema == schema

    def test_pydantic_model_schema(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        llm = ChatClaudeCode()
        new_llm = llm.with_structured_output(Person)
        assert new_llm._output_schema is not None
        assert new_llm._output_schema["type"] == "object"
        assert "name" in new_llm._output_schema["properties"]
        assert "age" in new_llm._output_schema["properties"]

    def test_invalid_type_raises_error(self):
        llm = ChatClaudeCode()
        with pytest.raises(TypeError, match="schema 必须是 dict 或 Pydantic 模型类"):
            llm.with_structured_output("not valid")

    def test_preserves_original_instance(self):
        llm = ChatClaudeCode()
        original_schema = llm._output_schema
        llm.with_structured_output({"type": "object"})
        assert llm._output_schema == original_schema


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode _generate() mock 测试
# ═══════════════════════════════════════════════════════════════


class TestGenerateWithMock:

    def test_normal_response(self, mock_subprocess_run):
        llm = ChatClaudeCode()
        result = llm._generate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "mocked response"
        assert llm._session_turn == 1

    def test_empty_messages(self):
        llm = ChatClaudeCode()
        result = llm._generate([])
        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == ""

    def test_timeout_error(self, mocker):
        mocker.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=300))
        llm = ChatClaudeCode()
        result = llm._generate([HumanMessage(content="hello")])
        assert "超时" in result.generations[0].message.content
        assert result.llm_output is not None
        assert result.llm_output.get("error") == "timeout"

    def test_file_not_found_error(self, mocker):
        mocker.patch("subprocess.run", side_effect=FileNotFoundError("claude not found"))
        llm = ChatClaudeCode()
        result = llm._generate([HumanMessage(content="hello")])
        assert "cli" in result.generations[0].message.content.lower() or \
               "PATH" in result.generations[0].message.content
        assert result.llm_output is not None
        assert result.llm_output.get("error") == "cli_not_found"

    def test_os_error(self, mocker):
        mocker.patch("subprocess.run", side_effect=OSError("permission denied"))
        llm = ChatClaudeCode()
        result = llm._generate([HumanMessage(content="hello")])
        assert "permission denied" in result.generations[0].message.content
        assert result.llm_output is not None
        assert result.llm_output.get("error") == "os_error"

    def test_command_error_returncode(self, mocker):
        """模拟 claude 返回非零 exit code"""
        mocker.patch("subprocess.run", return_value=FakeCompletedProcess(
            stdout="",
            stderr="claude: fatal error",
            returncode=1,
        ))
        llm = ChatClaudeCode()
        result = llm._generate([HumanMessage(content="hello")])
        assert "returncode=1" in result.generations[0].message.content


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode _stream() mock 测试
# ═══════════════════════════════════════════════════════════════


class TestStreamWithMock:

    def test_normal_stream(self, mock_subprocess_popen):
        llm = ChatClaudeCode()
        chunks = list(llm._stream([HumanMessage(content="hello")]))
        assert len(chunks) > 0
        assert llm._session_turn == 1
        # 验证产生了内容 chunk
        texts = [c for c in chunks if c.message.content]
        assert len(texts) >= 2  # "Hello" + " World"

    def test_empty_messages_stream(self):
        llm = ChatClaudeCode()
        chunks = list(llm._stream([]))
        assert len(chunks) == 1
        assert chunks[0].message.content == ""

    def test_file_not_found_stream(self, mocker):
        mocker.patch("subprocess.Popen", side_effect=FileNotFoundError("claude not found"))
        llm = ChatClaudeCode()
        chunks = list(llm._stream([HumanMessage(content="hello")]))
        assert len(chunks) == 1
        assert "cli" in chunks[0].message.content.lower()

    def test_os_error_stream(self, mocker):
        mocker.patch("subprocess.Popen", side_effect=OSError("permission"))
        llm = ChatClaudeCode()
        chunks = list(llm._stream([HumanMessage(content="hello")]))
        assert len(chunks) == 1
        assert "permission" in chunks[0].message.content

    def test_stop_word_triggers_terminate(self, mocker, mock_claude_stream_output):
        """验证 stop 词导致进程被 terminate"""
        mock_proc = FakePopen(stdout_lines=mock_claude_stream_output.split("\n"))
        mocker.patch("subprocess.Popen", return_value=mock_proc)
        llm = ChatClaudeCode()
        # "World" 作为 stop 词，在第二个 chunk 后触发终止
        list(llm._stream([HumanMessage(content="hello")], stop=["World"]))
        assert mock_proc.terminated


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode _agenerate() mock 测试 (async)
# ═══════════════════════════════════════════════════════════════


class TestAGenerateWithMock:

    @pytest.mark.asyncio
    async def test_normal_async(self, mocker):
        stdout = json.dumps({"result": "async response"}).encode("utf-8")
        mock_proc = FakeAsyncProcess(stdout_data=stdout, returncode=0)
        mocker.patch("asyncio.create_subprocess_exec", return_value=mock_proc)
        llm = ChatClaudeCode()
        result = await llm._agenerate([HumanMessage(content="hello")])
        assert isinstance(result, ChatResult)
        assert "async response" in result.generations[0].message.content
        assert llm._session_turn == 1

    @pytest.mark.asyncio
    async def test_async_timeout(self, mocker):
        import asyncio
        mocker.patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError)
        llm = ChatClaudeCode()
        result = await llm._agenerate([HumanMessage(content="hello")])
        assert result.llm_output is not None
        assert result.llm_output.get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_async_empty_messages(self):
        llm = ChatClaudeCode()
        result = await llm._agenerate([])
        assert result.generations[0].message.content == ""

    @pytest.mark.asyncio
    async def test_async_file_not_found(self, mocker):
        mocker.patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError)
        llm = ChatClaudeCode()
        result = await llm._agenerate([HumanMessage(content="hello")])
        assert result.llm_output is not None
        assert result.llm_output.get("error") == "cli_not_found"


# ═══════════════════════════════════════════════════════════════
# ChatClaudeCode _astream() mock 测试 (async)
# ═══════════════════════════════════════════════════════════════


class TestAStreamWithMock:

    @pytest.mark.asyncio
    async def test_normal_astream(self, mocker, mock_claude_stream_output):
        lines = [line.encode("utf-8") + b"\n" for line in mock_claude_stream_output.split("\n")]
        mock_proc = FakeAsyncProcess(stdout_lines=lines, returncode=0)

        async def _mock_subproc(*args, **kwargs):
            return mock_proc
        mocker.patch("asyncio.create_subprocess_exec", side_effect=_mock_subproc)
        llm = ChatClaudeCode()
        chunks = []
        async for chunk in llm._astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        assert len(chunks) > 0
        texts = [c for c in chunks if c.message.content]
        assert len(texts) >= 2

    @pytest.mark.asyncio
    async def test_astream_empty_messages(self):
        llm = ChatClaudeCode()
        chunks = []
        async for chunk in llm._astream([]):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].message.content == ""

    @pytest.mark.asyncio
    async def test_astream_file_not_found(self, mocker):
        async def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError
        mocker.patch("asyncio.create_subprocess_exec", side_effect=_raise_fnf)
        llm = ChatClaudeCode()
        chunks = []
        async for chunk in llm._astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "cli" in chunks[0].message.content.lower()

    @pytest.mark.asyncio
    async def test_astream_os_error(self, mocker):
        async def _raise_os(*args, **kwargs):
            raise OSError("test error")
        mocker.patch("asyncio.create_subprocess_exec", side_effect=_raise_os)
        llm = ChatClaudeCode()
        chunks = []
        async for chunk in llm._astream([HumanMessage(content="hello")]):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert "test error" in chunks[0].message.content


# ═══════════════════════════════════════════════════════════════
# _build_env() 测试
# ═══════════════════════════════════════════════════════════════


class TestBuildEnv:

    def test_basic_env_includes_os_environ(self):
        llm = ChatClaudeCode()
        env = llm._build_env()
        assert "PATH" in env

    def test_extra_env_merged(self):
        llm = ChatClaudeCode(extra_env={"MY_CUSTOM_VAR": "hello"})
        env = llm._build_env()
        assert env["MY_CUSTOM_VAR"] == "hello"

    def test_extra_env_overwrites(self, monkeypatch):
        monkeypatch.setenv("EXISTING_VAR", "original")
        llm = ChatClaudeCode(extra_env={"EXISTING_VAR": "overwritten"})
        env = llm._build_env()
        assert env["EXISTING_VAR"] == "overwritten"
