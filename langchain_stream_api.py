"""
LangChain 流式输出与交互式调整 FastAPI 服务
提供后端API接口供前端调用，支持思考过程区分显示
"""
from typing import List, Optional, Tuple
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

        # 兼容不同模型输出的思考标签
        self._think_open_tags = ("<think>", "<thinking>")
        self._think_close_tags = ("</think>", "</thinking>")

    @staticmethod
    def _split_by_first_tag(text: str, tags: Tuple[str, ...]) -> Tuple[str, Optional[str], str]:
        """按最先出现的标签切分文本，返回(标签前文本, 命中标签, 标签后文本)。"""
        first_pos = -1
        hit_tag = None
        for tag in tags:
            pos = text.find(tag)
            if pos != -1 and (first_pos == -1 or pos < first_pos):
                first_pos = pos
                hit_tag = tag

        if hit_tag is None:
            return text, None, ""

        return text[:first_pos], hit_tag, text[first_pos + len(hit_tag):]

    @staticmethod
    def _pending_suffix_len(text: str, tags: Tuple[str, ...]) -> int:
        """返回需要保留的尾部长度（可能是标签前缀，等待下一块内容补全）。"""
        if not text:
            return 0

        max_len = 0
        for tag in tags:
            upper = min(len(text), len(tag) - 1)
            for size in range(1, upper + 1):
                if text.endswith(tag[:size]):
                    max_len = max(max_len, size)
        return max_len


    @staticmethod
    def _format_sse_event(payload: str) -> str:
        """将任意 payload 编码为标准 SSE data 事件，避免内容中的\n\n破坏分帧。"""

        lines = payload.split("\n")
        return "".join(f"data: {line}\n" for line in lines) + "\n"

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
            类型：[THINKING] 统一流式内容（思考/回答都使用该标记）
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

        for chunk in self.llm.stream(formatted_messages):
            if chunk.content:
                content_buffer += chunk.content

                while content_buffer:
                    # 非思考阶段：寻找开始标签
                    if not in_thinking:
                        prefix, hit_tag, suffix = self._split_by_first_tag(content_buffer, self._think_open_tags)
                        if hit_tag is not None:
                            if prefix:
                                yield self._format_sse_event(f"[RESULT]{prefix}")
                            in_thinking = True
                            content_buffer = suffix
                            continue

                        pending_len = self._pending_suffix_len(content_buffer, self._think_open_tags)
                        if pending_len == 0:
                            yield self._format_sse_event(f"[RESULT]{content_buffer}")
                            content_buffer = ""
                        else:
                            flush_text = content_buffer[:-pending_len]
                            if flush_text:
                                yield self._format_sse_event(f"[RESULT]{flush_text}")
                            content_buffer = content_buffer[-pending_len:]
                        break

                    # 思考阶段：寻找结束标签
                    prefix, hit_tag, suffix = self._split_by_first_tag(content_buffer, self._think_close_tags)
                    if hit_tag is not None:
                        if show_thinking and prefix:
                            yield self._format_sse_event(f"[THINKING]{prefix}")
                        in_thinking = False
                        content_buffer = suffix
                        continue

                    pending_len = self._pending_suffix_len(content_buffer, self._think_close_tags)
                    if pending_len == 0:
                        if show_thinking and content_buffer:
                            yield self._format_sse_event(f"[THINKING]{content_buffer}")
                        content_buffer = ""
                    else:
                        flush_text = content_buffer[:-pending_len]
                        if show_thinking and flush_text:
                            yield self._format_sse_event(f"[THINKING]{flush_text}")
                        content_buffer = content_buffer[-pending_len:]
                    break

        # 处理缓冲区中剩余的内容（流式结束时的收尾）
        if content_buffer:
            if in_thinking:
                # 如果仍在思考状态（没有关闭标签），将剩余内容作为思考
                if show_thinking:
                    yield self._format_sse_event(f"[THINKING]{content_buffer}")
            else:
                # 非思考状态，按回答处理
                yield self._format_sse_event(f"[RESULT]{content_buffer}")


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
    - [THINKING]{content} - 统一流式内容（思考/结果都使用该标记）
    - [FULL_RESULT]{content} - 完整结果（流式结束时输出，基于原始 [RESULT] 内容）
    - [DONE] - 完成
    """
    session_id, session_data = session_manager.get_or_create_session(request.session_id)

    def generate():
        full_result = ""

        def _extract_sse_payload(event: str) -> str:
            data_parts = []
            for line in event.splitlines():
                if line.startswith("data:"):
                    data_parts.append(line[5:].lstrip())
            return "\n".join(data_parts)

        for token in llm_service.stream_generate(
            history=session_data["history"],
            user_input=request.message if not request.modifications else None,
            modifications=request.modifications,
            show_thinking=request.show_thinking
        ):
            stream_token = token
            if "[RESULT]" in stream_token:
                stream_token = stream_token.replace("[RESULT]", "[THINKING]", 1)
            yield stream_token
            # 收集结果内容（保持原始 [RESULT] 内容用于 FULL_RESULT）
            payload = _extract_sse_payload(token)
            if payload.startswith("[RESULT]"):
                full_result += payload[8:]
            elif payload.startswith("[FULL_RESULT]"):
                full_result = payload[13:]

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
        yield llm_service._format_sse_event(f"[FULL_RESULT]{full_result}")
        yield llm_service._format_sse_event("[DONE]")

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
