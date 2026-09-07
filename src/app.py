from dotenv import load_dotenv

load_dotenv()

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import asyncio
import html
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


def load_resource(*path_segments: str) -> str:
    """按相对路径加载前端资源文件 (CSS / JS / HTML)"""
    full_path = os.path.join(os.path.dirname(__file__), *path_segments)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


STATUS_BANNER_TEMPLATE = load_resource("templates", "status_banner.html")


def build_status_html(status_type: str, title: str, desc: str) -> str:
    """构建具有状态呼吸灯和高颜值现代卡片质感的状态 HTML"""
    icon_map = {"warning": "⚠️", "success": "✅", "danger": "❌", "info": "ℹ️"}
    icon = icon_map.get(status_type, "ℹ️")
    return STATUS_BANNER_TEMPLATE.format(
        status_type=status_type,
        icon=icon,
        title=title,
        desc=desc,
    )


async def predict(
    message: str, history: list, workspace_path: str = ""
) -> AsyncGenerator[str, None]:
    """Gradio 预测函数：必须指定项目工作区，流式输出 AI 响应及美化工具调用过程"""
    global _current_thread_id

    # 1. 严格检查工作区：未选定项目前拒绝提问并给出友好提示
    ws = (workspace_path or "").strip()
    if not ws or not os.path.exists(ws):
        yield (
            "⚠️ **请先在上方控制台绑定项目工作区**\n\n"
            "点击「📂 **浏览选择文件夹**」选择目标本地代码库，智能体将自动建立 RAG 代码符号知识库后即可开始对话与编码！"
        )
        return

    clean_msg = message.strip()
    # 当用户输入 /new 时清空上下文并重置会话
    if clean_msg.lower() in ("/new", "/clear", "/reset"):
        _current_thread_id = str(uuid.uuid4())
        yield "🔄 **会话已成功重置！** 历史上下文已清空，开启全新的代码工程交互。"
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

                escaped_name = html.escape(str(tool_name))
                escaped_args = html.escape(str(tool_args))
                escaped_output = html.escape(str(tool_output))

                # 美化工具调用展示：使用折叠卡片形式，防止大段工具返回污染对话视线
                tool_display = (
                    f'\n\n<details class="tool-call-card" open>\n'
                    f'<summary class="tool-call-summary">⚡ <strong>执行工具调用</strong> <code>{escaped_name}</code> <span class="tool-badge">已完成</span></summary>\n'
                    f'<div class="tool-call-body">\n'
                    f'<div class="tool-meta-label">📥 <strong>输入参数</strong></div>\n'
                    f'<pre class="tool-pre tool-args-pre"><code>{escaped_args}</code></pre>\n'
                    f'<div class="tool-meta-label">📤 <strong>执行结果</strong></div>\n'
                    f'<pre class="tool-pre tool-result-pre"><code>{escaped_output}</code></pre>\n'
                    f"</div>\n"
                    f"</details>\n\n"
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

        response += f"\n\n⚠️ **执行出错**: {e}"
        yield response

    finally:
        # 对话轮次结束后，在后台非阻塞执行当前工作区的增量指纹同步（若有文件变动则毫秒级热更新）
        asyncio.create_task(auto_sync_codebase(ws))

    # 兜底保障：若未产生任何输出，确保至少 yield 一次
    if not response:
        yield "（智能体已处理完毕，未返回进一步文本内容）"


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
        return current_path, build_status_html(
            "warning",
            "未选择有效文件夹",
            "请重新点击按钮并选择本地代码项目所在的根目录文件夹。",
        )

    print(f"\n[Workspace] 用户选定文件夹: {selected_dir}，开始执行索引同步...")

    try:
        sync_res = await auto_sync_codebase(selected_dir)
        status = sync_res.get("status")
        if status == "empty_workspace":
            detail = "当前工作区为空（未检测到代码文件）。已为你完成就绪，你可以直接向智能体提问，从零开始搭建和编写新项目！"
            await get_cached_agent(selected_dir)
            print(f"[Workspace] {detail}\n")
            return selected_dir, build_status_html("info", "工作区已就绪（空白项目）", detail)
        elif status == "full_indexed":
            detail = f"首次全量索引构建完成！共成功索引 {sync_res.get('total_files', 0)} 个工程代码文件。"
        elif status == "no_change":
            detail = f"代码索引校验通过：代码库无变更，智能体已热就绪（共 {sync_res.get('total_files', 0)} 个代码文件）。"
        elif status == "incrementally_synced":
            detail = f"增量同步完成！新增: {sync_res.get('added', 0)}, 修改: {sync_res.get('modified', 0)}, 删除: {sync_res.get('deleted', 0)}。"
        else:
            detail = f"索引处理完成：{sync_res}"

        # 预热并初始化该工作区 Agent
        await get_cached_agent(selected_dir)
        print(f"[Workspace] {detail}\n")
        return selected_dir, build_status_html("success", "代码库知识库已就绪", detail)
    except Exception as e:
        import traceback

        err_msg = f"索引构建失败: {str(e)}"
        print(f"[Workspace] {err_msg}")
        traceback.print_exc()
        return selected_dir, build_status_html("danger", "代码索引异常", err_msg)


async def handle_manual_sync(path: str) -> str:
    """手动点击或输入框修改后触发索引同步"""
    target = (path or "").strip()
    if not target or not os.path.exists(target):
        return build_status_html(
            "warning",
            "指定路径不存在",
            "请先输入或选择有效的本地项目路径后再进行同步。",
        )
    try:
        sync_res = await auto_sync_codebase(target)
        status = sync_res.get("status")
        if status == "empty_workspace":
            detail = "当前工作区为空（未检测到代码文件）。你可以直接向智能体提问，从零开始创建新项目与代码！"
            await get_cached_agent(target)
            return build_status_html("info", "工作区已就绪（空白项目）", detail)
        elif status == "full_indexed":
            detail = f"全量重新索引构建完成！共索引 {sync_res.get('total_files', 0)} 个代码文件。"
        elif status == "no_change":
            detail = f"索引无变更：当前代码库状态与知识库完全一致（共 {sync_res.get('total_files', 0)} 个文件）。"
        elif status == "incrementally_synced":
            detail = f"增量同步完成！新增: {sync_res.get('added', 0)}, 修改: {sync_res.get('modified', 0)}, 删除: {sync_res.get('deleted', 0)}。"
        else:
            detail = f"索引处理完成：{sync_res}"
        await get_cached_agent(target)
        return build_status_html("success", "知识库重新同步完成", detail)
    except Exception as e:
        return build_status_html("danger", "知识库同步失败", str(e))


# -------------------------------------------------------------
# 现代化 UI 主题与样式定义
# -------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.03)",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca 0%, #4f46e5 100%)",
    button_primary_text_color="#ffffff",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f8fafc",
    button_secondary_text_color="#334155",
    button_secondary_border_color="#cbd5e1",
)

# 加载外部前端静态资源 (CSS / JS / HTML Templates)
custom_css = load_resource("style.css")
script_js = load_resource("script.js")
fullscreen_head_html = f"<script>\n{script_js}\n</script>"

empty_state_html = load_resource("templates", "empty_state.html")
hero_card_html = load_resource("templates", "hero.html")

init_status_html = build_status_html(
    "warning",
    "尚未绑定项目工作区",
    "请点击上方「📂 浏览选择文件夹」绑定本地项目根目录后开始交互；支持在对话框中发送 /new 重置会话。",
)

# -------------------------------------------------------------
# 构建 Gradio Blocks 主应用
# -------------------------------------------------------------
with gr.Blocks(title="🤖 Code Agent - 全栈代码生成与 RAG 知识库", head=fullscreen_head_html) as demo:
    # 顶部 Hero Header 卡片
    gr.HTML(hero_card_html)

    # 工作区控制台面板
    with gr.Column(elem_classes=["console-card"]):
        gr.HTML(
            '<div class="console-header-label">📂 本地工作区控制台 (Workspace Console)</div>'
        )
        with gr.Row():
            workspace_input = gr.Textbox(
                show_label=False,
                value="",
                placeholder="请选择或粘贴本地代码库根目录绝对路径（如 C:/Users/zengd/Desktop/Snake）...",
                scale=7,
                container=False,
            )
            choose_btn = gr.Button(
                "📂 浏览选择文件夹",
                variant="primary",
                scale=2,
                elem_classes=["primary-btn-styled"],
            )
            sync_btn = gr.Button(
                "🔄 重新同步索引",
                variant="secondary",
                scale=1,
            )

        # 动态状态显示横幅
        status_display = gr.HTML(value=init_status_html)

    # 文件夹选择交互：唤起原生弹窗，选定后自动触发索引构建
    choose_btn.click(
        fn=handle_folder_selection_and_index,
        inputs=[workspace_input],
        outputs=[workspace_input, status_display],
    )

    # 手动同步按钮交互
    sync_btn.click(
        fn=handle_manual_sync,
        inputs=[workspace_input],
        outputs=[status_display],
    )

    # 快捷 Prompt 示例指令
    quick_examples = [
        ["🔍 分析当前项目的目录结构与核心业务架构", ""],
        ["🛡️ 检查代码库中的潜在逻辑漏洞与性能瓶颈", ""],
        ["✨ 帮我理清核心模块的业务流转过程与关键函数", ""],
        ["🧪 为核心模块编写清晰且高覆盖率的单元测试", ""],
    ]

    # 对话界面组件
    chat = gr.ChatInterface(
        fn=predict,
        additional_inputs=[workspace_input],
        chatbot=gr.Chatbot(
            height=540,
            placeholder=empty_state_html,
            buttons=["copy"],
            render_markdown=True,
            elem_id="main-chatbot",
        ),
        textbox=gr.Textbox(
            placeholder="💬 请输入您的开发需求、重构任务或代码提问（输入 /new 开启新会话）...",
            container=False,
            scale=8,
        ),
        examples=quick_examples,
        run_examples_on_click=False,
    )

    # 注入全屏控制脚本
    gr.HTML(fullscreen_head_html)

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=custom_css)
