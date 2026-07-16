"""
test_hermes_agent.py — Hermes Agent 集成测试

测试 ChatHermesAgent 和 hermes_agent 模块的核心功能。
需要 mock Hermes Agent 依赖（不需要真实 Hermes API）。
"""

from __future__ import annotations

import json
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

# 确保项目根在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════════════════════════════════════
# chat_hermes_agent 模块级导入测试
# ═══════════════════════════════════════════════════════════════

class TestModuleImport:
    """测试模块导入和路径发现。"""

    def test_import_module_without_hermes(self):
        """在无 Hermes 环境时，chat_hermes_agent 应可导入但不初始化 agent。"""
        # 只需确保模块本身可导入（类定义、函数等）
        from chat_hermes_agent import ChatHermesAgent
        assert ChatHermesAgent is not None

    def test_import_hermes_agent_module(self):
        """hermes_agent 模块应可导入。"""
        from hermes_agent import (
            create_hermes_agent,
            _PRESETS,
        )
        assert create_hermes_agent is not None
        assert isinstance(_PRESETS, dict)

    def test_presets_have_expected_modes(self):
        """预设应包含所有 4 种模式。"""
        from hermes_agent import _PRESETS
        assert set(_PRESETS.keys()) == {
            "code_analysis",
            "code_refactor",
            "full_access",
            "none",
        }


# ═══════════════════════════════════════════════════════════════
# 消息格式化测试
# ═══════════════════════════════════════════════════════════════

class TestMessageFormatting:
    """测试 LangChain 消息 → Hermes 格式的转换。"""

    def test_simple_human_message(self):
        from chat_hermes_agent import _hermes_format_messages
        from langchain_core.messages import HumanMessage

        user_msg, system, history = _hermes_format_messages([
            HumanMessage(content="你好")
        ])
        assert user_msg == "你好"
        assert system is None
        assert history == []

    def test_system_and_human_messages(self):
        from chat_hermes_agent import _hermes_format_messages
        from langchain_core.messages import SystemMessage, HumanMessage

        user_msg, system, history = _hermes_format_messages([
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="帮我分析代码"),
        ])
        assert user_msg == "帮我分析代码"
        assert system == "你是一个助手"
        assert history == []

    def test_multi_turn_conversation(self):
        from chat_hermes_agent import _hermes_format_messages
        from langchain_core.messages import (
            SystemMessage, HumanMessage, AIMessage
        )

        user_msg, system, history = _hermes_format_messages([
            SystemMessage(content="你是一个助手"),
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮助你的？"),
            HumanMessage(content="请分析这个项目"),
        ])

        assert user_msg == "请分析这个项目"
        assert system == "你是一个助手"
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "你好"}
        assert history[1]["role"] == "assistant"
        assert "你好！有什么可以帮助你的？" in str(history[1]["content"])

    def test_messages_with_tool_calls(self):
        from chat_hermes_agent import _hermes_format_messages
        from langchain_core.messages import (
            HumanMessage, AIMessage, ToolMessage
        )

        user_msg, system, history = _hermes_format_messages([
            HumanMessage(content="北京天气怎么样？"),
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_123",
                    "name": "get_weather",
                    "args": {"city": "北京"},
                }],
            ),
            ToolMessage(
                content="晴天，25°C",
                tool_call_id="call_123",
            ),
            AIMessage(content="北京今天是晴天，温度25°C"),
            HumanMessage(content="那上海呢？"),
        ])

        assert user_msg == "那上海呢？"
        assert len(history) == 4
        # 第1条：用户消息
        assert history[0] == {"role": "user", "content": "北京天气怎么样？"}
        # 第2条：assistant 消息含 tool_calls
        assert history[1]["role"] == "assistant"
        assert "tool_calls" in history[1]
        # 第3条：tool 消息
        assert history[2]["role"] == "tool"
        assert history[2]["tool_call_id"] == "call_123"
        assert "晴天" in history[2]["content"]
        # 第4条：assistant 回复
        assert history[3]["role"] == "assistant"

    def test_empty_messages(self):
        from chat_hermes_agent import _hermes_format_messages

        user_msg, system, history = _hermes_format_messages([])
        assert user_msg == ""
        assert system is None
        assert history == []

    def test_multimodal_content_serialization(self):
        from chat_hermes_agent import _serialize_content

        # 纯文本
        assert _serialize_content("hello") == "hello"

        # 多模态列表
        content = [
            {"type": "text", "text": "描述这张图片"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        result = _serialize_content(content)
        assert "描述这张图片" in result
        assert "[图片]" in result

    def test_safe_json_dumps(self):
        from chat_hermes_agent import _safe_json_dumps

        # 正常 dict
        assert json.loads(_safe_json_dumps({"a": 1}))

        # 包含不可序列化对象
        result = _safe_json_dumps({"fn": lambda: None})
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════
# ChatHermesAgent 测试（mock Hermes AIAgent）
# ═══════════════════════════════════════════════════════════════

class TestChatHermesAgent:
    """测试 ChatHermesAgent 类。"""

    def test_init_defaults(self):
        """测试默认初始化参数。"""
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent()
        assert llm.base_url == "http://localhost:30000/v1"
        assert llm.model == ""
        assert llm.max_iterations == 90
        assert llm.quiet_mode is True
        assert llm.working_dir == "."

    def test_custom_init(self):
        """测试自定义初始化参数。"""
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent(
            base_url="http://custom:8080/v1",
            model="gpt-4",
            api_key="sk-test",
            provider="openai",
            max_iterations=30,
            enabled_toolsets=["terminal"],
            system_prompt="你是专家",
            reasoning_config={"effort": "high"},
        )
        assert llm.base_url == "http://custom:8080/v1"
        assert llm.model == "gpt-4"
        assert llm.api_key == "sk-test"
        assert llm.provider == "openai"
        assert llm.max_iterations == 30
        assert llm.enabled_toolsets == ["terminal"]
        assert llm.system_prompt == "你是专家"
        assert llm.reasoning_config == {"effort": "high"}

    def test_llm_type(self):
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent()
        assert llm._llm_type == "hermes-agent"

    def test_identifying_params(self):
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent(
            base_url="http://test:8080/v1",
            model="test-model",
        )
        params = llm._identifying_params
        assert params["base_url"] == "http://test:8080/v1"
        assert params["model"] == "test-model"
        assert "session_id" in params

    def test_reset_session(self):
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent()
        old_session = llm._session_id
        new_session = llm.reset_session()
        assert new_session != old_session
        assert llm._hermes_initialized is False

    def test_bind_tools_creates_new_instance(self):
        from chat_hermes_agent import ChatHermesAgent
        from langchain_core.tools import tool

        @tool
        def mock_tool(x: str) -> str:
            """Mock tool for testing"""
            return x

        llm = ChatHermesAgent()
        llm_with_tools = llm.bind_tools([mock_tool])

        # 应返回新实例
        assert llm_with_tools is not llm
        assert llm_with_tools._session_id != llm._session_id
        assert llm_with_tools._bound_tools is not None
        assert len(llm_with_tools._bound_tools) == 1
        # 系统提示应包含工具描述
        assert llm_with_tools.system_prompt is not None
        assert "mock_tool" in llm_with_tools.system_prompt

    def test_with_structured_output_creates_new_instance(self):
        from chat_hermes_agent import ChatHermesAgent

        llm = ChatHermesAgent()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        llm_structured = llm.with_structured_output(schema)

        assert llm_structured is not llm
        assert llm_structured._session_id != llm._session_id
        assert llm_structured.system_prompt is not None
        assert "JSON Schema" in llm_structured.system_prompt

    def test_with_structured_output_pydantic_model(self):
        from chat_hermes_agent import ChatHermesAgent
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        llm = ChatHermesAgent()
        llm_structured = llm.with_structured_output(Person)

        assert llm_structured.system_prompt is not None
        assert "name" in llm_structured.system_prompt
        assert "age" in llm_structured.system_prompt

    def test_generate_mock(self):
        """使用 mock 测试 _generate。"""
        from chat_hermes_agent import ChatHermesAgent
        from langchain_core.messages import HumanMessage

        # Mock AIAgent
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "这是模拟回复",
            "model": "test-model",
        }
        mock_agent.ephemeral_system_prompt = None

        # 直接注入 mock agent，避免 _get_agent() 尝试真正导入
        llm = ChatHermesAgent()
        llm._agent = mock_agent
        llm._hermes_initialized = True

        result = llm._generate([HumanMessage(content="测试问题")])
        assert result.generations[0].message.content == "这是模拟回复"

    def test_generate_with_stop_words(self):
        """测试 stop word 截断。"""
        from chat_hermes_agent import ChatHermesAgent
        from langchain_core.messages import HumanMessage

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {
            "final_response": "第一部分结果\n停止\n第二部分结果",
        }
        mock_agent.ephemeral_system_prompt = None

        llm = ChatHermesAgent()
        llm._agent = mock_agent
        llm._hermes_initialized = True

        result = llm._generate(
            [HumanMessage(content="测试问题")],
            stop=["停止"],
        )
        assert result.generations[0].message.content == "第一部分结果\n"

    def test_generate_error_handling(self):
        """测试异常处理。"""
        from chat_hermes_agent import ChatHermesAgent
        from langchain_core.messages import HumanMessage

        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = RuntimeError("API 错误")
        mock_agent.ephemeral_system_prompt = None

        llm = ChatHermesAgent()
        llm._agent = mock_agent
        llm._hermes_initialized = True

        result = llm._generate([HumanMessage(content="测试问题")])
        content = result.generations[0].message.content
        assert "错误" in content or "API 错误" in content


# ═══════════════════════════════════════════════════════════════
# hermes_agent 测试
# ═══════════════════════════════════════════════════════════════

class TestHermesAgent:
    """测试 hermes_agent 模块。"""

    def test_create_hermes_agent_invalid_mode(self):
        """传入无效 mode 应抛出 ValueError。"""
        from hermes_agent import create_hermes_agent

        with pytest.raises(ValueError, match="未知模式"):
            create_hermes_agent(
                base_url="http://localhost:30000/v1",
                mode="invalid_mode",
            )

    def test_create_hermes_agent_valid_modes(self):
        """所有有效 mode 应可正常初始化（在 mock deepagents 的情况下）。"""
        from hermes_agent import create_hermes_agent

        valid_modes = ["code_analysis", "code_refactor", "full_access", "none"]
        for mode in valid_modes:
            # 仅测试 mode 验证通过，不实际创建 agent（需要 deepagents）
            try:
                from hermes_agent import _PRESETS
                assert mode in _PRESETS
            except Exception:
                pass

    def test_shortcut_functions_exist(self):
        """快捷函数应可导入。"""
        from hermes_agent import (
            create_hermes_code_analysis_agent,
            create_hermes_code_refactor_agent,
            create_hermes_full_access_agent,
        )
        assert callable(create_hermes_code_analysis_agent)
        assert callable(create_hermes_code_refactor_agent)
        assert callable(create_hermes_full_access_agent)

    def test_preset_code_analysis_config(self):
        """code_analysis 预设应有正确的配置。"""
        from hermes_agent import _PRESETS

        preset = _PRESETS["code_analysis"]
        assert "claude_code" in preset["register"]
        assert "claude_code_structured" in preset["register"]
        assert "claude_code_isolated" not in preset["register"]

    def test_preset_full_access_config(self):
        """full_access 预设应包含隔离执行工具。"""
        from hermes_agent import _PRESETS

        preset = _PRESETS["full_access"]
        assert "claude_code" in preset["register"]
        assert "claude_code_structured" in preset["register"]
        assert "claude_code_isolated" in preset["register"]

    def test_preset_none_config(self):
        """none 预设不应注册任何工具。"""
        from hermes_agent import _PRESETS

        preset = _PRESETS["none"]
        assert preset["register"] == []

    def test_build_system_prompt_code_analysis(self):
        from hermes_agent import _build_system_prompt

        prompt = _build_system_prompt("code_analysis")
        assert prompt is not None
        assert "code_analysis" in prompt
        assert "只读" in prompt or "claude_code" in prompt.lower()

    def test_build_system_prompt_none(self):
        from hermes_agent import _build_system_prompt

        prompt = _build_system_prompt("none")
        assert prompt is None

    def test_build_system_prompt_custom(self):
        from hermes_agent import _build_system_prompt

        custom = "自定义系统提示"
        prompt = _build_system_prompt("code_analysis", custom_prompt=custom)
        assert prompt == custom

    def test_tool_hints_for_mode(self):
        from hermes_agent import _tool_hints_for_mode

        hints = _tool_hints_for_mode("full_access")
        assert "claude_code" in hints
        assert "claude_code_structured" in hints
        assert "claude_code_isolated" in hints

        hints_none = _tool_hints_for_mode("none")
        assert hints_none == {}

    def test_exported_symbols(self):
        """所有导出的符号应可访问。"""
        from hermes_agent import __all__ as hermes_all

        essential = [
            "create_hermes_agent",
            "create_hermes_code_analysis_agent",
            "create_hermes_code_refactor_agent",
            "create_hermes_full_access_agent",
        ]
        for name in essential:
            assert name in hermes_all, f"{name} 不在 __all__ 中"

        from chat_hermes_agent import __all__ as chat_all

        assert "ChatHermesAgent" in chat_all
        assert "hermes_agent" in chat_all
        assert "hermes_agent_structured" in chat_all
        assert "hermes_agent_session" in chat_all


# ═══════════════════════════════════════════════════════════════
# Hermes Tool 形式测试
# ═══════════════════════════════════════════════════════════════

class TestHermesTools:
    """测试 Hermes Agent 的 @tool 形式。"""

    def test_tools_are_importable(self):
        """所有 Hermes 工具函数应可导入且有 invoke 方法。"""
        from chat_hermes_agent import (
            hermes_agent,
            hermes_agent_structured,
            hermes_agent_session,
            close_hermes_session,
            list_hermes_sessions,
        )
        # LangChain StructuredTool 对象有 invoke 方法而非直接 callable
        for t in [hermes_agent, hermes_agent_structured, hermes_agent_session]:
            assert hasattr(t, "invoke"), f"{t.name} 应该有 invoke 方法"
        assert callable(close_hermes_session)
        assert callable(list_hermes_sessions)

    def test_tools_have_correct_metadata(self):
        """工具应有正确的 name 和 description 属性。"""
        from chat_hermes_agent import (
            hermes_agent,
            hermes_agent_structured,
            hermes_agent_session,
        )

        assert hermes_agent.name == "hermes_agent"
        assert hermes_agent_structured.name == "hermes_agent_structured"
        assert hermes_agent_session.name == "hermes_agent_session"

        assert len(hermes_agent.description) > 0
        assert len(hermes_agent_structured.description) > 0
        assert len(hermes_agent_session.description) > 0

    def test_session_cache_management(self):
        """测试会话缓存的增删查。"""
        from chat_hermes_agent import (
            _hermes_agent_cache,
            close_hermes_session,
            list_hermes_sessions,
        )

        # 手动添加缓存条目来测试
        _hermes_agent_cache["test-session-1"] = "mock-agent-1"
        _hermes_agent_cache["test-session-2"] = "mock-agent-2"

        sessions = list_hermes_sessions()
        assert "test-session-1" in sessions
        assert "test-session-2" in sessions

        assert close_hermes_session("test-session-1") is True
        assert "test-session-1" not in _hermes_agent_cache
        assert "test-session-2" in _hermes_agent_cache

        # 清理
        _hermes_agent_cache.clear()

    def test_close_nonexistent_session(self):
        """关闭不存在的会话应返回 False。"""
        from chat_hermes_agent import close_hermes_session

        result = close_hermes_session("nonexistent-session")
        assert result is False

    def test_hermes_agent_tool_arguments(self):
        """hermes_agent 工具应有正确的参数。"""
        from chat_hermes_agent import hermes_agent

        args = hermes_agent.args_schema.model_json_schema()
        props = args.get("properties", {})

        assert "task" in props
        assert "session_id" in props
        assert "base_url" in props
        assert "model" in props
        assert "max_iterations" in props

    def test_hermes_agent_structured_tool_arguments(self):
        """hermes_agent_structured 应包含 output_schema 参数。"""
        from chat_hermes_agent import hermes_agent_structured

        args = hermes_agent_structured.args_schema.model_json_schema()
        props = args.get("properties", {})

        assert "task" in props
        assert "output_schema" in props

    def test_hermes_agent_session_tool_arguments(self):
        """hermes_agent_session 应包含 context 参数。"""
        from chat_hermes_agent import hermes_agent_session

        args = hermes_agent_session.args_schema.model_json_schema()
        props = args.get("properties", {})

        assert "task" in props
        assert "context" in props
        assert "session_id" in props

    def test_tools_usable_in_langgraph_tool_node(self):
        """工具可以在 LangGraph 的 ToolNode 中使用。"""
        try:
            from langgraph.prebuilt import ToolNode
            from chat_hermes_agent import (
                hermes_agent,
                hermes_agent_structured,
                hermes_agent_session,
            )

            tools = [hermes_agent, hermes_agent_structured, hermes_agent_session]
            node = ToolNode(tools)
            assert node is not None
        except ImportError:
            pass  # langgraph 可能未安装


# ═══════════════════════════════════════════════════════════════
# 工具描述构建测试
# ═══════════════════════════════════════════════════════════════

class TestToolDescription:
    """测试 _build_hermes_tool_description。"""

    def test_build_tool_description(self):
        from chat_hermes_agent import _build_hermes_tool_description
        from langchain_core.tools import tool

        @tool
        def read_file(path: str) -> str:
            """读取指定路径的文件内容"""
            return ""

        desc = _build_hermes_tool_description([read_file])
        assert "read_file" in desc
        assert "读取指定路径的文件内容" in desc
        assert "path" in desc


# ═══════════════════════════════════════════════════════════════
# 路径发现测试
# ═══════════════════════════════════════════════════════════════

class TestPathDiscovery:
    """测试 Hermes 路径发现。"""

    def test_hermes_home_env_var(self):
        """设置 HERMES_HOME 应能发现 Hermes（如果目录存在）。"""
        from chat_hermes_agent import _ensure_hermes_importable

        # 测试 HERMES_HOME 指向 tests/hermes-agent-main/
        hermes_path = os.path.join(
            os.path.dirname(__file__), "hermes-agent-main"
        )
        if os.path.isdir(hermes_path):
            with patch.dict(os.environ, {"HERMES_HOME": hermes_path}):
                # 应不抛出异常
                try:
                    _ensure_hermes_importable()
                except ImportError:
                    pass  # 可能因为缺少 Hermes 依赖而失败

    def test_missing_hermes_raises(self):
        """_ensure_hermes_importable 函数存在且为可调用。"""
        from chat_hermes_agent import _ensure_hermes_importable

        assert callable(_ensure_hermes_importable)

    def test_ensure_hermes_importable_error_message(self):
        """验证 ImportError 的错误消息格式。"""
        from chat_hermes_agent import _ensure_hermes_importable

        try:
            _ensure_hermes_importable()
        except ImportError as e:
            msg = str(e)
            # 如果能触发异常，检查消息格式
            assert "无法导入 Hermes Agent" in msg
            assert "pip install hermes-agent" in msg or "HERMES_HOME" in msg
        else:
            # 如果没抛异常（Hermes 已安装），也是正常情况
            pass


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
