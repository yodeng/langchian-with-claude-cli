"""ChatClaudeCode 与 Deep Agent 的 FastAPI 接口

提供两种后端：
  - /chat       — ChatClaudeCode 简单对话（流式 + 非流式）
  - /deep-agent — Deep Agent 编排执行（流式 + 非流式）

启动:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

ChatClaudeCode 调用:
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "分析当前目录下的项目结构"}'

Deep Agent 调用:
    curl -X POST http://localhost:8000/deep-agent \
      -H "Content-Type: application/json" \
      -d '{"message": "分析项目架构"}'
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_claude_code import ChatClaudeCode
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger("api_server")

app = FastAPI(
    title="Claude Code API",
    description="ChatClaudeCode 与 Deep Agent 的 REST API 接口",
    version="1.0.0",
)

# ═══════════════════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """ChatClaudeCode 对话请求"""

    message: str = Field(..., description="用户消息内容")
    session_id: str | None = Field(
        default=None,
        description="会话 ID，不传则自动创建新会话",
    )
    working_dir: str = Field(default=".", description="工作目录")
    effort: str = Field(
        default="medium",
        description="努力级别: low/medium/high/xhigh/max",
    )
    system_prompt: str | None = Field(default=None, description="系统提示词")
    timeout: int = Field(
        default=600,
        ge=10,
        le=3600,
        description="超时秒数",
    )
    model: str | None = Field(default=None, description="模型名称")


class DeepAgentRequest(BaseModel):
    """Deep Agent 编排请求"""

    message: str = Field(..., description="用户消息/任务描述")
    mode: str = Field(
        default="code_analysis",
        description="预设模式: code_analysis/code_refactor/full_access/none",
    )
    model: str | None = Field(
        default=None,
        description="编排层 LLM，默认 deepseek-v4-pro",
    )
    claude_tool_effort: str | None = Field(
        default=None,
        description="Claude Code 工具 effort 级别",
    )
    system_prompt: str | None = Field(default=None, description="自定义系统提示词")


class ChatResponse(BaseModel):
    """非流式对话响应"""

    content: str = Field(..., description="回复内容")
    session_id: str = Field(..., description="会话 ID")
    usage: dict[str, Any] | None = Field(default=None, description="token 用量")


class DeepAgentResponse(BaseModel):
    """Deep Agent 响应"""

    content: str = Field(..., description="最终结果内容")
    mode: str = Field(..., description="使用的模式")


# ═══════════════════════════════════════════════════════════════
# Session 管理（仅 /chat 端点用）
# ═══════════════════════════════════════════════════════════════

_sessions: dict[str, ChatClaudeCode] = {}


def _get_or_create_llm(request: ChatRequest) -> tuple[str, ChatClaudeCode]:
    """获取或创建会话对应的 ChatClaudeCode 实例。"""
    if request.session_id and request.session_id in _sessions:
        return request.session_id, _sessions[request.session_id]

    sid = request.session_id or str(uuid.uuid4())
    llm = ChatClaudeCode(
        working_dir=request.working_dir,
        effort=request.effort,
        system_prompt=request.system_prompt,
        timeout=request.timeout,
        model=request.model,
    )
    _sessions[sid] = llm
    return sid, llm


# ═══════════════════════════════════════════════════════════════
# /chat — ChatClaudeCode 简单对话
# ═══════════════════════════════════════════════════════════════


@app.get("/health")
def health():
    """健康检查"""
    return {"status": "ok", "active_sessions": len(_sessions)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """非流式对话。"""
    sid, llm = _get_or_create_llm(request)

    try:
        result = llm.invoke([HumanMessage(content=request.message)])
    except Exception as e:
        logger.error("非流式调用失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        content=str(result.content),
        session_id=sid,
        usage=getattr(result, "usage_metadata", None),
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """流式对话（SSE）。"""
    sid, llm = _get_or_create_llm(request)

    def event_generator():
        try:
            for chunk in llm.stream([HumanMessage(content=request.message)]):
                if chunk.content:
                    yield f"data: {json.dumps({'type': 'delta', 'content': chunk.content})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': sid})}\n\n"
        except Exception as e:
            logger.error("流式调用失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════
# /deep-agent — Deep Agent 编排执行
# ═══════════════════════════════════════════════════════════════


@app.post("/deep-agent", response_model=DeepAgentResponse)
def deep_agent(request: DeepAgentRequest):
    """非流式 Deep Agent 编排。

    使用 deep_agent 进行任务规划、分解和执行。
    """
    try:
        from deep_agent import create_claude_deep_agent
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="需要安装 deepagents: pip install -e '.[deepagent]'",
        )

    try:
        agent = create_claude_deep_agent(
            model=request.model,
            mode=request.mode,
            system_prompt=request.system_prompt,
            claude_tool_effort=request.claude_tool_effort,
        )
        result = agent.invoke({
            "messages": [{"role": "user", "content": request.message}],
        })
    except Exception as e:
        logger.error("Deep Agent 调用失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # 提取最后一条 AI 消息
    messages = result.get("messages", [])
    last_ai = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) or (
            isinstance(m, dict) and m.get("type") == "ai"
        ):
            content = m.content if hasattr(m, "content") else m.get("content", "")
            if content:
                last_ai = content
                break

    return DeepAgentResponse(content=str(last_ai), mode=request.mode)


@app.post("/deep-agent/stream")
def deep_agent_stream(request: DeepAgentRequest):
    """流式 Deep Agent 编排（SSE）。

    实时推送 deep_agent 编排过程中产生的消息。
    """
    try:
        from deep_agent import create_claude_deep_agent
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="需要安装 deepagents: pip install -e '.[deepagent]'",
        )

    def event_generator():
        try:
            agent = create_claude_deep_agent(
                model=request.model,
                mode=request.mode,
                system_prompt=request.system_prompt,
                claude_tool_effort=request.claude_tool_effort,
            )
            for chunk, _metadata in agent.stream(
                {"messages": [{"role": "user", "content": request.message}]},
                stream_mode="messages",
            ):
                content = chunk.content if hasattr(chunk, "content") else ""
                if content and isinstance(content, str):
                    yield f"data: {json.dumps({'type': 'delta', 'content': content})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'mode': request.mode})}\n\n"
        except Exception as e:
            logger.error("Deep Agent 流式调用失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════
# 会话管理（仅 /chat 端点）
# ═══════════════════════════════════════════════════════════════


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """删除指定会话，释放资源。"""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/sessions")
def list_sessions():
    """列出当前活跃会话。"""
    return {
        "count": len(_sessions),
        "sessions": list(_sessions.keys()),
    }
