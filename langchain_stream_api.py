"""
LangChain 流式输出与交互式调整 FastAPI 服务
提供后端API接口供前端调用，支持思考过程区分显示
"""
from typing import List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_openai import ChatOpenAI

# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    session_id: Optional[str] = None
    message: str
    modifications: Optional[List[str]] = None
    show_thinking: bool = True


class ResetRequest(BaseModel):
    """重置请求模型"""
    session_id: str


# ==================== 会话管理 ====================

class SessionManager:
    """会话管理器，维护多用户的对话历史"""

    def __init__(self):
        self.sessions = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple:
        """获取或创建会话"""
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]

        new_id = session_id or str(uuid4())
        self.sessions[new_id] = {"history": [], "modifications": []}
        return new_id, self.sessions[new_id]

    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话数据"""
        return self.sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"] = []
            self.sessions[session_id]["modifications"] = []
            return True
        return False


# 全局会话管理器
session_manager = SessionManager()
# ==================== LLM 配置 ====================

class LangChainLLMService:
    """LangChain LLM服务"""

    def __init__(self):
        # self.llm = ChatTongyi(
        #     api_key='sk-bb5511a45a1a4613a531ff77e37e9ffc',
        #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     model="qwen-plus",
        #     max_tokens=4000,
        #     model_kwargs={"enable_thinking": True}
        # )

        self.llm = ChatOpenAI(
            api_key='sk-no-key-required',
            base_url="http://172.16.140.213:9080/qwen32/v1",
            model="gpt-3.5-turbo",
            max_tokens=4000,
        )

    def stream_generate(
        self,
        history: List,
        user_input: Optional[str] = None,
        modifications: Optional[List[str]] = None,
        show_thinking: bool = True
    ):
        """
        流式生成内容，支持思考过程区分显示

        Yields:
            流式输出的token，格式：data: [type]content\n\n
            类型：[THINKING] 思考内容, [RESULT] 回答内容
        """
        messages = list(history)

        # 构建完整输入
        if user_input:
            input_text = user_text = user_input
        elif modifications:
            input_text = "根据以下指令调整之前的回答：\n"
            for i, mod in enumerate(modifications, 1):
                input_text += f"{i}. {mod}\n"
        else:
            return

        # 转换为消息对象
        formatted_messages = []
        for m in messages:
            if hasattr(m, 'type') and hasattr(m, 'content'):
                if not m.content or not m.content.strip():
                    continue
                if m.type == "ai":
                    formatted_messages.append(AIMessage(content=m.content))
                else:
                    formatted_messages.append(HumanMessage(content=m.content))
        formatted_messages.append(HumanMessage(content=input_text))

        # 流式生成
        content_buffer = ""  # 用于缓冲 content，处理思考内容分割
        in_thinking = False  # 是否在思考标签内
        thinking_buffer = ""  # 思考内容缓冲区

        for chunk in self.llm.stream(formatted_messages):
            if chunk.content:
                content_buffer += chunk.content

                # 检查 <thinking> 标签
                if "<thinking>" in content_buffer and not in_thinking:
                    # 发现 <thinking> 标签，进入思考模式
                    parts = content_buffer.split("<thinking>", 1)
                    # 前面的部分是回答（如果有）
                    if parts[0]:
                        yield f"data: [RESULT]{parts[0]}|||SSE|||"
                    in_thinking = True
                    thinking_buffer = ""
                    content_buffer = parts[1] if len(parts) > 1 else ""
                    continue

                # 检查 </thinking> 标签
                if "</thinking>" in content_buffer and in_thinking:
                    # 发现 </thinking> 标签，结束思考模式
                    parts = content_buffer.split("</thinking>", 1)
                    thinking_buffer += parts[0]

                    # 发送完整的思考内容
                    if show_thinking and thinking_buffer:
                        yield f"data: [THINKING]{thinking_buffer}|||SSE|||"

                    in_thinking = False
                    thinking_buffer = ""
                    content_buffer = parts[1] if len(parts) > 1 else ""
                    continue

                # 根据当前状态处理内容
                if in_thinking:
                    # 在思考标签内，继续累积思考内容
                    thinking_buffer += content_buffer
                    content_buffer = ""
                else:
                    # 在回答阶段，发送结果内容
                    yield f"data: [RESULT]{content_buffer}|||SSE|||"
                    content_buffer = ""

        # 处理缓冲区中剩余的内容（流式结束时的收尾）
        if in_thinking and (thinking_buffer or content_buffer):
            # 如果仍在思考状态（没有 </thinking>），将所有内容作为思考
            if show_thinking:
                yield f"data: [THINKING]{thinking_buffer + content_buffer}|||SSE|||"
        elif content_buffer:
            # 否则作为回答
            yield f"data: [RESULT]{content_buffer}|||SSE|||"


llm_service = LangChainLLMService()


# ==================== FastAPI 应用 ====================

app = FastAPI(title="LangChain Stream API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 路由 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "LangChain Stream API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/api/chat/stream",
            "reset": "/api/reset",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口，支持思考过程和结果区分显示

    SSE格式说明：
    - [THINKING]{content} - 思考内容
    - [RESULT]{content} - 结果内容
    - [FULL_RESULT]{content} - 完整结果（流式结束时输出）
    - [DONE] - 完成
    """
    session_id, session_data = session_manager.get_or_create_session(request.session_id)

    def generate():
        full_result = ""
        thinking_result = ""  # 收集完整的思考内容

        for token in llm_service.stream_generate(
            history=session_data["history"],
            user_input=request.message if not request.modifications else None,
            modifications=request.modifications,
            show_thinking=request.show_thinking
        ):
            yield token
            # 收集结果内容
            if "[THINKING]" in token:
                start_idx = token.find("[THINKING]")
                end_idx = token.find("|||SSE|||", start_idx)
                if start_idx != -1 and end_idx != -1:
                    thinking_result += token[start_idx + 10:end_idx]
            elif "[RESULT]" in token:
                start_idx = token.find("[RESULT]")
                end_idx = token.find("|||SSE|||", start_idx)
                if start_idx != -1 and end_idx != -1:
                    full_result += token[start_idx + 8:end_idx]
            elif "[FULL_RESULT]" in token:
                start_idx = token.find("[FULL_RESULT]")
                end_idx = token.find("|||SSE|||", start_idx)
                if start_idx != -1 and end_idx != -1:
                    full_result = token[start_idx + 12:end_idx]

        # 保存对话历史
        if request.modifications:
            if session_data["history"] and isinstance(session_data["history"][-1], AIMessage):
                session_data["history"].pop()
            prompt = "根据以下指令调整之前的回答：\n"
            for i, mod in enumerate(request.modifications, 1):
                prompt += f"{i}. {mod}\n"
            session_data["history"].append(HumanMessage(content=prompt))
            session_data["modifications"] = request.modifications
        else:
            session_data["history"].append(HumanMessage(content=request.message))
            session_data["modifications"] = []

        session_data["history"].append(AIMessage(content=full_result))
        yield f"data: [FULL_RESULT]{full_result}|||SSE|||"
        yield "data: [DONE]|||SSE|||"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/reset")
async def reset_session(request: ResetRequest):
    """重置会话历史"""
    success = session_manager.clear_session(request.session_id)
    if success:
        return {"status": "success", "message": "会话已重置"}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """获取会话信息"""
    session = session_manager.get_session(session_id)
    if session:
        return {
            "session_id": session_id,
            "message_count": len(session["history"]) // 2,
            "modifications": session["modifications"]
        }
    raise HTTPException(status_code=404, detail="会话不存在")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
