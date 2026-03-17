"""
LangChain Stream API 测试客户端
使用 Python 调用 FastAPI 服务，支持思考过程区分显示
"""
import requests
import uuid
from typing import Optional

API_BASE = "http://localhost:8000"


def test_health():
    """测试健康检查"""
    print("\n=== 健康检查 ===")
    response = requests.get(f"{API_BASE}/api/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")


def chat_stream(
    session_id: str,
    message: str = None,
    modifications: list = None,
    show_thinking: bool = True
):
    """
    流式聊天，支持思考过程区分显示

    Args:
        session_id: 会话ID
        message: 用户消息
        modifications: 调整指令列表
        show_thinking: 是否显示思考过程
    """
    print("\n" + "=" * 60)
    if message:
        print(f"用户: {message}")
    elif modifications:
        print(f"调整指令: {', '.join(modifications)}")

    response = requests.post(
        f"{API_BASE}/api/chat/stream",
        json={
            "session_id": session_id,
            "message": message,
            "modifications": modifications,
            "show_thinking": show_thinking
        },
        stream=True
    )

    result_content = ""
    thinking_content = ""  # 用于收集完整的思考内容
    in_thinking = False
    in_result = False


    # 流式读取字节流，使用自定义分隔符
    buffer = ""
    for chunk in response.iter_content(chunk_size=1024, decode_unicode=False):
        if chunk:

            buffer += chunk.decode('utf-8', errors='ignore')

            # 按 |||SSE||| 分割消息（自定义分隔符）
            while "|||SSE|||" in buffer:
                message, buffer = buffer.split("|||SSE|||", 1)

                # 移除 "data: " 前缀
                if message.startswith("data: "):
                    data = message[6:]
                elif message.startswith("data:"):
                    data = message[5:]
                else:
                    continue  # 跳过非 data: 开头的消息


                # 处理各种标记
                if data == "[DONE]":
                    break

                # 处理思考内容
                if data.startswith("[THINKING]"):
                    content = data[10:]
                    thinking_content += content
                    if show_thinking:
                        print(content, end='', flush=True)
                    continue

                # 处理回答内容
                if data.startswith("[RESULT]"):
                    content = data[8:]
                    result_content += content
                    print(content, end='', flush=True)
                    continue

                # 处理完整回答
                if data.startswith("[FULL_RESULT]"):
                    result_content = data[len("[FULL_RESULT]"):]
                    print("\n" + "-" * 60)
                    print("📋 完整回答:")
                    print("-" * 60)
                    print(result_content)
                    print("-" * 60)
                    # 打印摘要信息
                    print(f"DEBUG: think 长度: {len(thinking_content)} 字符")
                    print(f"DEBUG: content 长度: {len(result_content)} 字符")
                    continue

    print()
    return result_content


def reset_session(session_id: str):
    """重置会话"""
    print("\n=== 重置会话 ===")
    response = requests.post(
        f"{API_BASE}/api/reset",
        json={"session_id": session_id}
    )
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")


def get_session_info(session_id: str):
    """获取会话信息"""
    print(f"\n=== 会话信息: {session_id} ===")
    response = requests.get(f"{API_BASE}/api/session/{session_id}")
    if response.status_code == 200:
        info = response.json()
        print(f"消息数量: {info['message_count']}")
        print(f"调整指令: {', '.join(info['modifications']) if info['modifications'] else '无'}")
    else:
        print(f"错误: {response.status_code}")


def interactive_demo(show_thinking: bool = True):
    """交互式演示"""
    print("=" * 60)
    print("LangChain Stream API 测试客户端")
    print(f"显示思考过程: {'开启' if show_thinking else '关闭'}")
    print("=" * 60)
    print(f"\nAPI 地址: {API_BASE}")
    print("=" * 60)

    session_id = str(uuid.uuid4())
    print(f"\n创建会话: {session_id}")

    chat_stream(session_id, "介绍一下Python", show_thinking=show_thinking)
    get_session_info(session_id)
    chat_stream(session_id, modifications=["更详细一点"], show_thinking=show_thinking)
    get_session_info(session_id)
    chat_stream(session_id, modifications=["加上应用场景"], show_thinking=show_thinking)
    get_session_info(session_id)
    chat_stream(session_id, "什么是LangChain?", show_thinking=show_thinking)
    get_session_info(session_id)
    reset_session(session_id)
    chat_stream(session_id, "重置后的第一个问题", show_thinking=show_thinking)
    get_session_info(session_id)

    print("\n" + "=" * 60)
    print("演示结束")
    print("=" * 60)


def manual_mode():
    """手动模式"""
    print("=" * 60)
    print("LangChain Stream API - 手动模式")
    print("=" * 60)

    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("\n错误: 无法连接到 API 服务")
        print("请先运行: python langchain_stream_api.py")
        return

    show_thinking = input("\n是否显示思考过程? (y/n, 默认y): ").lower() != 'n'

    session_id = str(uuid.uuid4())
    print(f"\n会话ID: {session_id}")
    print("输入 'exit' 退出，'reset' 重置会话，'info' 查看会话信息")
    print("输入 'toggle' 切换思考过程显示\n")

    modifications = []

    while True:
        print("\n" + "-" * 60)
        cmd = input("请输入 (消息/调整指令/命令): ").strip()

        if not cmd:
            continue

        if cmd.lower() == 'exit':
            break

        if cmd.lower() == 'reset':
            reset_session(session_id)
            modifications = []
            session_id = str(uuid.uuid4())
            print(f"新会话ID: {session_id}")
            continue

        if cmd.lower() == 'info':
            get_session_info(session_id)
            continue

        if cmd.lower() == 'toggle':
            show_thinking = not show_thinking
            print(f"思考过程显示已{'开启' if show_thinking else '关闭'}")
            continue

        is_modification = input("这是调整指令吗? (y/n): ").lower() == 'y'

        if is_modification:
            modifications.append(cmd)
            chat_stream(session_id, modifications=modifications, show_thinking=show_thinking)
        else:
            modifications = []
            chat_stream(session_id, message=cmd, show_thinking=show_thinking)

        get_session_info(session_id)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            show_thinking = len(sys.argv) < 3 or sys.argv[2] != "--no-thinking"
            interactive_demo(show_thinking)
        elif sys.argv[1] == "--help":
            print("""
                用法:
                python test_client.py                  # 手动模式（默认显示思考过程）
                python test_client.py --demo            # 演示模式（显示思考过程）
                python test_client.py --demo --no-thinking  # 演示模式（不显示思考过程）

                命令:
                exit        - 退出程序
                reset       - 重置会话
                info        - 查看会话信息
                toggle      - 切换思考过程显示
                <消息>      - 发送普通消息
                <调整指令>  - 发送调整指令
            """)
    else:
        manual_mode()
