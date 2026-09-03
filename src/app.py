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


def build_status_html(status_type: str, title: str, desc: str) -> str:
    """构建具有状态呼吸灯和高颜值现代卡片质感的状态 HTML"""
    icon_map = {"warning": "⚠️", "success": "✅", "danger": "❌", "info": "ℹ️"}
    icon = icon_map.get(status_type, "ℹ️")
    return f"""
<div class="status-banner status-{status_type}">
    <div class="status-indicator">
        <span class="pulse-dot dot-{status_type}"></span>
        <span class="status-icon">{icon}</span>
    </div>
    <div class="status-content">
        <div class="status-title">{title}</div>
        <div class="status-desc">{desc}</div>
    </div>
</div>
"""


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

                # 美化工具调用展示：使用折叠卡片形式，防止大段工具返回污染对话视线
                tool_display = (
                    f'\n\n<details class="tool-call-card" open>\n'
                    f'<summary class="tool-call-summary">⚡ <strong>执行工具调用</strong> <code>{tool_name}</code> <span class="tool-badge">已完成</span></summary>\n'
                    f'<div class="tool-call-body">\n'
                    f'<div class="tool-meta-label">📥 <strong>输入参数</strong></div>\n'
                    f'<pre class="tool-pre"><code>{tool_args}</code></pre>\n'
                    f'<div class="tool-meta-label">📤 <strong>执行结果</strong></div>\n'
                    f'<pre class="tool-pre tool-result-pre"><code>{tool_output}</code></pre>\n'
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
        if status == "full_indexed":
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
        err_msg = f"索引构建失败: {str(e)}"
        print(f"[Workspace] {err_msg}")
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
        if status == "full_indexed":
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

custom_css = """
/* 全局容器最大宽度与居中 */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 24px 20px 48px 20px !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 顶部品牌卡片 */
.hero-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8faff 50%, #f1f5ff 100%);
    border: 1px solid #e0e7ff;
    border-radius: 18px;
    padding: 24px 28px;
    margin-bottom: 20px;
    box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.08), 0 8px 10px -6px rgba(79, 70, 229, 0.04);
}

.hero-header-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.hero-avatar-box {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    flex-shrink: 0;
}

.hero-title-group h1 {
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
    background: linear-gradient(120deg, #1e293b 0%, #312e81 60%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    margin: 4px 0 0 0;
    font-size: 13.5px;
    color: #64748b;
    font-weight: 400;
}

.hero-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.tech-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    color: #475569;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    transition: all 0.2s ease;
}

.tech-pill:hover {
    border-color: #cbd5e1;
    transform: translateY(-1px);
}

.pill-indigo {
    background: #eef2ff;
    border-color: #c7d2fe;
    color: #4338ca;
}

.pill-emerald {
    background: #ecfdf5;
    border-color: #a7f3d0;
    color: #065f46;
}

/* 控制台卡片区域 */
.console-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 18px 22px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
}

.console-header-label {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #64748b;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* 状态通知横幅 */
.status-banner {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 12px;
    margin-top: 14px;
    transition: all 0.3s ease;
}

.status-warning {
    background: #fffbeb;
    border: 1px solid #fef3c7;
    color: #92400e;
}

.status-success {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
}

.status-danger {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.status-indicator {
    display: flex;
    align-items: center;
    position: relative;
    padding-top: 2px;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    position: relative;
}

.dot-warning {
    background: #f59e0b;
    box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
    animation: pulse-warn 2s infinite;
}

.dot-success {
    background: #10b981;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    animation: pulse-succ 2s infinite;
}

.dot-danger {
    background: #ef4444;
}

@keyframes pulse-warn {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(245, 158, 11, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 158, 11, 0); }
}

@keyframes pulse-succ {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

.status-content {
    flex: 1;
}

.status-title {
    font-size: 13.5px;
    font-weight: 600;
    line-height: 1.4;
}

.status-desc {
    font-size: 12.5px;
    opacity: 0.9;
    margin-top: 2px;
    line-height: 1.4;
}

/* 按钮微动效 */
.primary-btn-styled {
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px rgba(79, 70, 229, 0.25) !important;
    transition: all 0.2s ease !important;
}

.primary-btn-styled:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 15px rgba(79, 70, 229, 0.35) !important;
}

/* 工具调用折叠卡片 */
.tool-call-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 12px 0;
    font-size: 13px;
}

.tool-call-summary {
    cursor: pointer;
    font-weight: 600;
    color: #334155;
    outline: none;
    display: flex;
    align-items: center;
    gap: 8px;
}

.tool-call-summary code {
    background: #e0e7ff;
    color: #4338ca;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
}

.tool-badge {
    background: #ecfdf5;
    color: #047857;
    font-size: 11px;
    padding: 1px 8px;
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid #a7f3d0;
}

.tool-call-body {
    margin-top: 10px;
    border-top: 1px dashed #cbd5e1;
    padding-top: 8px;
}

.tool-meta-label {
    font-size: 11.5px;
    font-weight: 600;
    color: #64748b;
    margin: 6px 0 3px 0;
}

.tool-pre {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    overflow-x: auto !important;
    max-height: 160px !important;
    overflow-y: auto !important;
    margin: 4px 0 !important;
}

.tool-result-pre {
    max-height: 220px !important;
}

/* 空状态欢迎区 */
.empty-hero-box {
    text-align: center;
    padding: 36px 16px;
    color: #475569;
}

.empty-icon {
    font-size: 42px;
    margin-bottom: 12px;
    animation: float-anim 3s ease-in-out infinite;
}

@keyframes float-anim {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}

.empty-title {
    font-size: 18px;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 6px;
}

.empty-desc {
    font-size: 13.5px;
    color: #64748b;
    max-width: 520px;
    margin: 0 auto 20px auto;
    line-height: 1.5;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
    max-width: 680px;
    margin: 0 auto;
    text-align: left;
}

.feature-item {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 12.5px;
    color: #334155;
    display: flex;
    align-items: center;
    gap: 8px;
}

.feature-item span {
    font-size: 16px;
}

/* 聊天框阴影与圆角微调 */
#main-chatbot {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.04) !important;
}
"""

empty_state_html = """
<div class="empty-hero-box">
    <div class="empty-icon">🤖</div>
    <div class="empty-title">Code Agent 全栈代码开发就绪</div>
    <div class="empty-desc">
        请在上方控制台选定您的本地代码工程文件夹，系统将秒级挂载本地 RAG 知识库与 LangGraph 自主工具链。
    </div>
    <div class="feature-grid">
        <div class="feature-item">
            <span>🗺️</span>
            <div><strong>架构感知</strong><br><small style="color:#64748b">全局符号与依赖结构扫描</small></div>
        </div>
        <div class="feature-item">
            <span>🔍</span>
            <div><strong>精准 RAG 检索</strong><br><small style="color:#64748b">Chroma 向量与语义关联</small></div>
        </div>
        <div class="feature-item">
            <span>🛠️</span>
            <div><strong>多工具协作</strong><br><small style="color:#64748b">自主调用文件与代码分析工具</small></div>
        </div>
        <div class="feature-item">
            <span>🔄</span>
            <div><strong>增量热同步</strong><br><small style="color:#64748b">实时比对哈希，自动感知代码变更</small></div>
        </div>
    </div>
</div>
"""

init_status_html = build_status_html(
    "warning",
    "尚未绑定项目工作区",
    "请点击上方「📂 浏览选择文件夹」绑定本地项目根目录后开始交互；支持在对话框中发送 /new 重置会话。",
)

# -------------------------------------------------------------
# 构建 Gradio Blocks 主应用
# -------------------------------------------------------------
with gr.Blocks(title="🤖 Code Agent - 全栈代码生成与 RAG 知识库") as demo:
    # 顶部 Hero Header 卡片
    gr.HTML(
        """
    <div class="hero-card">
        <div class="hero-header-row">
            <div class="hero-brand">
                <div class="hero-avatar-box">⚡</div>
                <div class="hero-title-group">
                    <h1>Code Agent</h1>
                    <div class="hero-subtitle">全栈代码工程智能体 · 具备 RAG 知识库与 LangGraph 自主工具协作</div>
                </div>
            </div>
            <div class="hero-pills">
                <span class="tech-pill pill-indigo">⚙️ LangGraph StateEngine</span>
                <span class="tech-pill pill-emerald">📚 Chroma 向量 RAG</span>
                <span class="tech-pill">🔄 增量指纹同步</span>
                <span class="tech-pill">🛠️ 自主工具闭环</span>
            </div>
        </div>
    </div>
    """
    )

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

if __name__ == "__main__":
    demo.launch(theme=custom_theme, css=custom_css)
