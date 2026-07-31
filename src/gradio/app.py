import gradio as gr
from agent.index import get_agent
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, ToolMessage
from typing import AsyncGenerator

# 全局保存 agent 实例
_agent: CompiledStateGraph | None = None


async def get_cached_agent() -> CompiledStateGraph:
    global _agent
    if _agent is None:
        _agent = await get_agent()
    return _agent


async def predict(message: str, history: list) -> AsyncGenerator[str, None]:
    """Gradio 预测函数：通过 LangGraph astream 异步流式输出 AI 响应及工具调用过程"""
    agent = await get_cached_agent()

    # 配置 LangGraph 的 thread_id 保持对话历史
    config: RunnableConfig = {
        "configurable": {"thread_id": "gradio_default_session"},
    }

    response = ""
    current_tool_name = ""
    current_tool_args = ""

    try:
        async for chunk, metadata in agent.astream(
            {"messages": [("user", message)]},
            config=config,
            stream_mode="messages",
        ):
            node = (
                metadata.get("langgraph_node") if isinstance(metadata, dict) else None
            )

            # 1. 捕获模型发起的工具调用请求及参数
            if (
                node in ("model", "agent")
                and hasattr(chunk, "tool_call_chunks")
                and chunk.tool_call_chunks
            ):
                for tc in chunk.tool_call_chunks:
                    if isinstance(tc, dict):
                        name = tc.get("name")
                        args = tc.get("args")
                        if name:
                            current_tool_name = name
                            current_tool_args = ""
                        if args:
                            current_tool_args += args

            # 2. 捕获工具节点的输出结果 (node == "tools" 或 ToolMessage)
            elif node == "tools" or isinstance(chunk, ToolMessage):
                tool_name = getattr(chunk, "name", "") or current_tool_name or "tool"
                tool_args = current_tool_args.strip() or "{}"
                tool_output = str(chunk.content)

                tool_display = (
                    f"\n\n> 🛠️ **调用工具**: `{tool_name}`\n"
                    f"> 📥 **参数**: `{tool_args}`\n"
                    f"> 📤 **结果**: `{tool_output}`\n\n---\n\n"
                )
                response += tool_display
                current_tool_name = ""
                current_tool_args = ""
                yield response

            # 3. 捕获模型的正常文本输出 (排除 ToolMessage)
            elif (
                node in ("model", "agent")
                and isinstance(chunk, BaseMessage)
                and chunk.content
            ):
                if not isinstance(chunk, ToolMessage):
                    token = (
                        chunk.content
                        if isinstance(chunk.content, str)
                        else str(chunk.content)
                    )
                    if token:
                        response += token
                        yield response

    except Exception as e:
        response += f"\n\n⚠️ 发生错误: {e}"
        yield response

    # 兜底保障：若未产生任何输出，确保至少 yield 一次，避免 Gradio 报 StopAsyncIteration
    if not response:
        yield "（模型未返回任何内容）"


# 使用 Gradio ChatInterface 搭建聊天界面
demo = gr.ChatInterface(
    fn=predict,
    title="🤖 AI 助手",
    description="👋 你好！我是你的 AI 助手。",
    textbox=gr.Textbox(placeholder="请输入您的消息...", container=False, scale=7),
)

if __name__ == "__main__":
    demo.launch()
