import chainlit as cl
from typing import cast
from agent.index import get_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage


# ==========================================
# 2. Chainlit 生命周期：会话初始化
# ==========================================
@cl.on_chat_start
async def on_chat_start():
    agent = await get_agent()
    # 将 agent 和 thread_id 保存到 Chainlit 的 User Session 中
    cl.user_session.set("agent", agent)
    # 使用 Chainlit 的 session_id 作为 LangGraph 的 thread_id
    thread_id = cl.user_session.get("id")
    cl.user_session.set("thread_id", thread_id)

    # 发送欢迎消息
    await cl.Message(content="👋 你好！我是你的 AI 助手。").send()


# ==========================================
# 3. Chainlit 生命周期：处理用户消息与流式响应
# ==========================================
@cl.on_message
async def on_message(message: cl.Message):
    # 从 session 中提取配置
    agent = cast(CompiledStateGraph, cl.user_session.get("agent"))
    thread_id = cl.user_session.get("thread_id")

    # 创建一个空的 Chainlit 消息容器用于流式打字效果
    msg = cl.Message(content="")

    # 使用 LangchainCallbackHandler 将 Tool 调用和思考过程实时渲染在 UI 上
    cb = cl.LangchainCallbackHandler(stream_final_response=True)

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [cb],
    }

    # 异步流式调用 Agent
    async for chunk in agent.astream(
        {"messages": [("user", message.content)]},
        config=config,
        stream_mode="messages",
    ):
        # chunk[0] 是消息对象，chunk[1] 是 metadata
        message_chunk, metadata = chunk

        # 只处理模型最终输出的内容（过滤掉 Tool 的中间消息）
        if (
            isinstance(message_chunk, BaseMessage)
            and message_chunk.content
            and isinstance(metadata, dict)
            and metadata.get("langgraph_node") == "agent"
        ):
            token = (
                message_chunk.content
                if isinstance(message_chunk.content, str)
                else str(message_chunk.content)
            )
            await msg.stream_token(token)

    # 发送完毕，结束消息流
    await msg.send()
