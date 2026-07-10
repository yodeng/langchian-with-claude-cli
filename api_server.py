"""ChatClaudeCode FastAPI 接口

提供流式和非流式两种调用方式。

启动:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

非流式调用:
    curl -X POST http://localhost:8000/chat \
      -H "Content-Type: application/json" \
      -d '{"message": "分析当前目录下的项目结构"}'

流式调用:
    curl -X POST http://localhost:8000/chat/stream \
      -H "Content-Type: application/json" \
      -d '{"message": "解释这段代码的作用"}' \
      --no-buffer
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
from langchain_core.messages import HumanMessage

logger = logging.getLogger("api_server")

app = FastAPI(
    title="Claude Code API",
    description="ChatClaudeCode 的 REST API 接口，支持流式和非流式调用",
    version="1.0.0",
)

# ═══════════════════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """通用对话请求"""

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


class ChatResponse(BaseModel):
    """非流式对话响应"""

    content: str = Field(..., description="回复内容")
    session_id: str = Field(..., description="会话 ID")
    usage: dict[str, Any] | None = Field(default=None, description="token 用量")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    active_sessions: int


# ═══════════════════════════════════════════════════════════════
# Session 管理
# ═══════════════════════════════════════════════════════════════

_sessions: dict[str, ChatClaudeCode] = {}


def _get_or_create_llm(request: ChatRequest) -> tuple[str, ChatClaudeCode]:
    """获取或创建会话对应的 ChatClaudeCode 实例。

    - 请求带 session_id 且存在 → 复用已有实例（保持上下文）
    - 否则 → 创建新实例
    """
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
# 端点
# ═══════════════════════════════════════════════════════════════


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return HealthResponse(status="ok", active_sessions=len(_sessions))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """非流式对话。

    发送一条消息，返回完整回复。会话上下文通过 session_id 保持。
    """
    sid, llm = _get_or_create_llm(request)

    try:
        result = llm.invoke([HumanMessage(content=request.message)])
    except Exception as e:
        logger.error("非流式调用失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    usage = getattr(result, "usage_metadata", None)

    return ChatResponse(
        content=str(result.content),
        session_id=sid,
        usage=usage,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式对话（SSE）。

    发送一条消息，通过 SSE 实时推送生成的文本片段。
    """
    sid, llm = _get_or_create_llm(request)

    async def event_generator():
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


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话，释放资源。"""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/sessions")
async def list_sessions():
    """列出当前活跃会话。"""
    return {
        "count": len(_sessions),
        "sessions": list(_sessions.keys()),
    }
