from dotenv import load_dotenv
load_dotenv()

import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import asyncio
import uuid
from typing import AsyncGenerator

import gradio as gr
from agent.index import get_agent
from tools.rag_tools import auto_sync_codebase
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, ToolMessage

# 全局保存 agent 实例、当前工作区与当前会话 ID
_agent: CompiledStateGraph | None = None
_current_workspace: str | None = None
_current_thread_id: str = str(uuid.uuid4())


async def get_cached_agent(workspace_path: str) -> CompiledStateGraph:
    """按工作区路径懒加载并缓存 Agent 实例"""
    global _agent, _current_workspace, _current_thread_id
    ws = os.path.abspath(workspace_path.strip())

    # 如果工作区发生变化，重新创建 Agent 并开启新会话
    if _agent is None or _current_workspace != ws:
        print(f"[Workspace] 切换工作区: {_current_workspace} -> {ws}")
        _current_workspace = ws
        _agent = await get_agent(ws)
        _current_thread_id = str(uuid.uuid4())

    return _agent


async def predict(
    message: str, history: list, workspace_path: str = ""
) -> AsyncGenerator[str, None]:
    """Gradio 预测函数：必须指定项目工作区，流式输出 AI 响应及工具调用过程"""
    global _current_thread_id

    # 1. 严格检查工作区：未选定项目前拒绝提问并给出友好提示
    ws = (workspace_path or "").strip()
    if not ws or not os.path.exists(ws):
        yield "⚠️ **请先在上方点击「📂 浏览选择文件夹」选择或输入本地项目路径**，然后即可开始提问与代码生成！"
        return

    clean_msg = message.strip()
    # 当用户输入 /new 时清空上下文并重置会话
    if clean_msg.lower() in ("/new", "/clear", "/reset"):
        _current_thread_id = str(uuid.uuid4())
        yield "🔄 **会话已成功重置！** 历史上下文已清空，开启全新的对话。"
        return

    agent = await get_cached_agent(ws)

    # 配置 LangGraph 的 thread_id 保持多轮对话历史
    config: RunnableConfig = {
        "configurable": {"thread_id": _current_thread_id},
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
        import traceback
        print(f"[ERROR] 发生异常: {e}")
        traceback.print_exc()
        
        response += f"\n\n⚠️ 发生错误: {e}"
        yield response

    finally:
        # 对话轮次结束后，在后台非阻塞执行当前工作区的增量指纹同步（若有文件变动则毫秒级热更新）
        asyncio.create_task(auto_sync_codebase(ws))

    # 兜底保障：若未产生任何输出，确保至少 yield 一次，避免 Gradio 报 StopAsyncIteration
    if not response:
        yield "（模型未返回任何内容）"


def open_folder_dialog(current_path: str) -> str:
    """唤起系统原生文件夹选择对话框并返回真实绝对路径"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # 保持弹窗置顶
        initial_dir = (
            current_path
            if (current_path and os.path.exists(current_path))
            else os.path.expanduser("~")
        )
        selected_dir = filedialog.askdirectory(
            initialdir=initial_dir, title="选择项目工作区文件夹"
        )
        root.destroy()
        return selected_dir if selected_dir else current_path
    except Exception as e:
        print(f"[警告] 唤起文件夹选择对话框失败: {e}")
        return current_path


async def handle_folder_selection_and_index(current_path: str) -> tuple[str, str]:
    """选择文件夹并在选定后立即触发该项目的索引构建/指纹增量同步"""
    selected_dir = open_folder_dialog(current_path)
    if not selected_dir or not os.path.exists(selected_dir):
        return current_path, "⚠️ 未选择有效文件夹"

    print(f"\n[Workspace] 用户选定文件夹: {selected_dir}，开始执行索引同步...")

    try:
        sync_res = await auto_sync_codebase(selected_dir)
        status = sync_res.get("status")
        if status == "full_indexed":
            detail = f"✅ 首次全量索引构建完成！共索引 {sync_res.get('total_files', 0)} 个代码文件。"
        elif status == "no_change":
            detail = f"✅ 索引校验通过：代码库无变更，跳过索引（共 {sync_res.get('total_files', 0)} 个文件）。"
        elif status == "incrementally_synced":
            detail = f"✅ 增量同步完成！新增: {sync_res.get('added')}, 修改: {sync_res.get('modified')}, 删除: {sync_res.get('deleted')}。"
        else:
            detail = f"✅ 索引处理完成：{sync_res}"

        # 预热并初始化该工作区 Agent
        await get_cached_agent(selected_dir)
        print(f"[Workspace] {detail}\n")
        return selected_dir, detail
    except Exception as e:
        err_msg = f"❌ 索引构建失败: {str(e)}"
        print(f"[Workspace] {err_msg}")
        return selected_dir, err_msg


async def handle_manual_sync(path: str) -> str:
    """手动点击或输入框修改后触发索引同步"""
    target = (path or "").strip()
    if not target or not os.path.exists(target):
        return f"⚠️ 请先指定有效的项目路径！"
    try:
        sync_res = await auto_sync_codebase(target)
        status = sync_res.get("status")
        if status == "full_indexed":
            detail = f"✅ 首次全量索引构建完成！共索引 {sync_res.get('total_files', 0)} 个代码文件。"
        elif status == "no_change":
            detail = f"✅ 索引校验通过：代码库无变更，跳过索引（共 {sync_res.get('total_files', 0)} 个文件）。"
        elif status == "incrementally_synced":
            detail = f"✅ 增量同步完成！新增: {sync_res.get('added')}, 修改: {sync_res.get('modified')}, 删除: {sync_res.get('deleted')}。"
        else:
            detail = f"✅ 索引处理完成：{sync_res}"
        await get_cached_agent(target)
        return detail
    except Exception as e:
        return f"❌ 索引同步失败: {str(e)}"


# 使用 Gradio Blocks 构建界面
with gr.Blocks(title="🤖 Code Agent (具备 RAG 代码知识库与专家技能)") as demo:
    gr.Markdown("# 🤖 Code Agent (全栈代码生成与 RAG 知识库)")
    gr.Markdown(
        "👋 你好！我是全栈代码智能体。请先在下方点击 **「📂 浏览选择文件夹」** 绑定项目工程，系统将自动构建/增量同步代码知识库。"
    )

    with gr.Row():
        workspace_input = gr.Textbox(
            label="📁 当前项目工作区根目录 (Workspace Path)",
            value="",
            placeholder="请先点击右侧按钮选择本地代码库文件夹（如 C:/Users/zengd/Desktop/Snake）...",
            scale=7,
        )
        choose_btn = gr.Button("📂 浏览选择文件夹", variant="primary", scale=2)
        sync_btn = gr.Button("🔄 重新同步索引", scale=1)

    status_display = gr.Markdown(
        value="⚠️ **当前尚未选定项目**：请先点击上方按钮选择目标代码工程文件夹后再开始提问。",
    )

    # 点击选择文件夹按钮后，唤起弹窗并在选定后立即触发该工作区的索引构建
    choose_btn.click(
        fn=handle_folder_selection_and_index,
        inputs=[workspace_input],
        outputs=[workspace_input, status_display],
    )

    # 手动同步按钮
    sync_btn.click(
        fn=handle_manual_sync,
        inputs=[workspace_input],
        outputs=[status_display],
    )

    # 聊天界面组件
    chat = gr.ChatInterface(
        fn=predict,
        additional_inputs=[workspace_input],
        textbox=gr.Textbox(
            placeholder="请输入您的开发需求或代码提问（发送 /new 开启新会话）...",
            container=False,
            scale=7,
        ),
    )

if __name__ == "__main__":
    demo.launch()
