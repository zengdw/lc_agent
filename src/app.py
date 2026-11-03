from dotenv import load_dotenv

load_dotenv()

import os

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# 自动确保当前运行环境具备 pip 模块
try:
    import pip  # noqa: F401
except ImportError:
    try:
        import ensurepip
        ensurepip.bootstrap(upgrade=True)
    except Exception:
        pass


import asyncio
import gradio as gr

from agent.session_manager import (
    create_session,
    list_sessions,
    get_session,
    delete_session,
)
from ui.resources import load_resource
from ui.state import app_state
from ui.workspace import (
    get_workspace_status_for_session,
    handle_folder_selection_and_index,
    handle_manual_sync,
)
from ui.session_view import (
    get_session_messages_for_chatbot,
    render_session_list_html,
    get_session_workflow,
)
from ui.design_panel import (
    load_workflow_panels_for_session,
    get_workflow_panel_data,
)
from ui.model_config_ui import build_model_config_ui
from ui.predict_handler import predict


# 初始化会话与状态
_initial_chatbot_messages = app_state.bootstrap()

# 现代化 UI 主题与样式定义
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

_init_ws_val, init_status_html = get_workspace_status_for_session(
    app_state.current_workspace or ""
)
_init_design_html, _init_test_cases_md, _init_test_report_md = asyncio.run(
    load_workflow_panels_for_session(app_state.current_thread_id)
)

# -------------------------------------------------------------
# 构建 Gradio Blocks 主应用
# -------------------------------------------------------------
with gr.Blocks(title="🤖 Code Agent - 全栈工程智能体 (多 Agent 协作)") as demo:
    # 顶部 Hero Header 卡片（已内嵌模型配置按钮）
    gr.HTML(hero_card_html, elem_classes=["hero-wrapper-block"])

    # 隐藏的弹窗触发桥梁按钮
    open_model_modal_btn = gr.Button(
        elem_id="open-model-modal-btn",
        elem_classes=["hidden-action-trigger"],
    )

    # 模态弹窗：多 Agent 独立模型配置
    modal_container = build_model_config_ui()
    open_model_modal_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[modal_container],
    )

    # 左侧会话历史侧边栏（可折叠，默认展开）
    with gr.Sidebar(label="会话管理", position="left", open=True, width=280):
        new_chat_btn = gr.Button(
            "➕ 新建会话",
            variant="primary",
            elem_classes=["primary-btn-styled", "new-chat-btn"],
        )
        with gr.Row(elem_classes=["session-list-header-row"]):
            gr.HTML('<div class="session-list-header-title">历史会话列表</div>')

        session_list_html = gr.HTML(
            value=render_session_list_html(app_state.current_thread_id),
            elem_id="custom-session-container",
        )

    # 右侧流程产物侧边栏（类似会话列表，默认折叠收缩）
    with gr.Sidebar(label="🎯 流程产物预览 (设计与测试)", position="right", open=False, width=440, elem_classes=["artifacts-sidebar"]):
        with gr.Tabs(elem_classes=["artifacts-sidebar-tabs"]):
            with gr.TabItem("🎨 设计稿预览"):
                design_preview_display = gr.HTML(
                    value=_init_design_html,
                    elem_id="design-preview-container",
                )
            with gr.TabItem("📋 测试用例"):
                test_cases_display = gr.Markdown(
                    value=_init_test_cases_md,
                    elem_id="test-cases-container",
                )
            with gr.TabItem("🧪 测试报告"):
                test_report_display = gr.Markdown(
                    value=_init_test_report_md,
                    elem_id="test-report-container",
                )

    # 中间主工作区：紧凑型工作区控制台 + 对话界面
    with gr.Column(elem_classes=["main-chat-container"]):
        # 紧凑单行工作区控制台
        with gr.Column(elem_classes=["console-card-compact"]):
            with gr.Row(elem_classes=["console-compact-row"]):
                gr.HTML('<span class="console-label-tag">📂 工作区</span>', scale=0)
                workspace_input = gr.Textbox(
                    show_label=False,
                    value=_init_ws_val,
                    placeholder="请选择本地代码工程路径...",
                    scale=6,
                    container=False,
                    elem_classes=["compact-ws-input"],
                )
                choose_btn = gr.Button(
                    "📂 浏览选择",
                    variant="primary",
                    scale=2,
                    elem_classes=["compact-btn-primary"],
                )
                sync_btn = gr.Button(
                    "🔄 重新同步",
                    variant="secondary",
                    scale=1,
                    elem_classes=["compact-btn-secondary"],
                )

            # 紧凑型状态横幅
            status_display = gr.HTML(value=init_status_html, elem_classes=["compact-status-box"])

        # 文件夹选择交互
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

        # 居中占满高度的 ChatInterface 对话区
        with gr.Column(elem_id="chat-interface-wrapper"):
            chat = gr.ChatInterface(
                fn=predict,
                additional_inputs=[workspace_input],
                additional_outputs=[design_preview_display, test_cases_display, test_report_display],
                chatbot=gr.Chatbot(
                    value=_initial_chatbot_messages,
                    height=520,
                    placeholder=empty_state_html,
                    buttons=["copy"],
                    render_markdown=True,
                    elem_id="main-chatbot",
                ),
                textbox=gr.Textbox(
                    placeholder="💬 请输入您的开发需求、架构任务或代码提问（输入 /new 开启新会话）...",
                    container=False,
                    scale=8,
                    submit_btn=True,
                    stop_btn=True,
                    elem_id="main-chatbot-input-box",
                ),
                submit_btn=True,
                stop_btn=True,
                run_examples_on_click=False,
            )

    # 隐藏的跨前端后端通信桥梁
    with gr.Row(elem_id="session-hidden-bridge-container"):
        target_select_input = gr.Textbox(
            elem_id="target-select-session-id",
            elem_classes=["hidden-action-trigger"],
        )
        target_select_btn = gr.Button(
            elem_id="target-select-btn",
            elem_classes=["hidden-action-trigger"],
        )
        target_delete_input = gr.Textbox(
            elem_id="target-delete-session-id",
            elem_classes=["hidden-action-trigger"],
        )
        target_delete_btn = gr.Button(
            elem_id="target-delete-btn",
            elem_classes=["hidden-action-trigger"],
        )

    # 会话管理交互事件绑定
    async def on_select_session(target_sid: str):
        """用户点击切换会话：联动同步消息历史、工作区状态与右侧产物面板"""
        sid = (target_sid or "").strip()
        if not sid:
            sid = app_state.current_thread_id
        else:
            app_state.current_thread_id = sid

        msgs = await get_session_messages_for_chatbot(app_state.current_thread_id)
        sess = get_session(app_state.current_thread_id)
        ws_path = sess.get("workspace_path", "") if sess else ""
        ws_val, status_html = get_workspace_status_for_session(ws_path)

        if ws_val and os.path.exists(ws_val):
            app_state.current_workspace = ws_val
            asyncio.create_task(app_state.get_cached_agent(ws_val))
        else:
            app_state.current_workspace = None

        d_html, tc_md, tr_md = await load_workflow_panels_for_session(app_state.current_thread_id)

        return (
            render_session_list_html(app_state.current_thread_id),
            msgs,
            msgs,
            ws_val,
            status_html,
            d_html,
            tc_md,
            tr_md,
        )

    target_select_btn.click(
        fn=on_select_session,
        inputs=[target_select_input],
        outputs=[
            session_list_html,
            chat.chatbot,
            chat.chatbot_state,
            workspace_input,
            status_display,
            design_preview_display,
            test_cases_display,
            test_report_display,
        ],
        js="() => [window.__target_select_session_id || '']",
    )

    async def on_new_chat(current_input_ws: str):
        """点击新建会话按钮：继承当前工作区并清空右侧产物面板"""
        ws_to_bind = (current_input_ws or app_state.current_workspace or "").strip()
        app_state.current_thread_id = create_session(title="新会话", workspace_path=ws_to_bind)
        ws_val, status_html = get_workspace_status_for_session(ws_to_bind)
        if ws_val and os.path.exists(ws_val):
            app_state.current_workspace = ws_val

        d_html, tc_md, tr_md = await load_workflow_panels_for_session(app_state.current_thread_id)
        return (
            render_session_list_html(app_state.current_thread_id),
            [],
            [],
            ws_val,
            status_html,
            d_html,
            tc_md,
            tr_md,
        )

    new_chat_btn.click(
        fn=on_new_chat,
        inputs=[workspace_input],
        outputs=[
            session_list_html,
            chat.chatbot,
            chat.chatbot_state,
            workspace_input,
            status_display,
            design_preview_display,
            test_cases_display,
            test_report_display,
        ],
    )

    async def on_delete_specific_session(target_sid: str):
        """删除指定 ID 的会话"""
        sid_to_delete = target_sid.strip() if target_sid else ""
        if sid_to_delete:
            delete_session(sid_to_delete)

        remaining = list_sessions()
        if remaining:
            remaining_ids = [s["session_id"] for s in remaining]
            if (
                app_state.current_thread_id == sid_to_delete
                or app_state.current_thread_id not in remaining_ids
            ):
                app_state.current_thread_id = remaining[0]["session_id"]
            msgs = await get_session_messages_for_chatbot(app_state.current_thread_id)
        else:
            app_state.current_thread_id = create_session(
                title="新会话", workspace_path=app_state.current_workspace or ""
            )
            msgs = []

        sess = get_session(app_state.current_thread_id)
        ws_path = sess.get("workspace_path", "") if sess else ""
        ws_val, status_html = get_workspace_status_for_session(ws_path)
        if ws_val and os.path.exists(ws_val):
            app_state.current_workspace = ws_val
        else:
            app_state.current_workspace = None

        d_html, tc_md, tr_md = await load_workflow_panels_for_session(app_state.current_thread_id)

        return (
            render_session_list_html(app_state.current_thread_id),
            msgs,
            msgs,
            ws_val,
            status_html,
            d_html,
            tc_md,
            tr_md,
        )

    target_delete_btn.click(
        fn=on_delete_specific_session,
        inputs=[target_delete_input],
        outputs=[
            session_list_html,
            chat.chatbot,
            chat.chatbot_state,
            workspace_input,
            status_display,
            design_preview_display,
            test_cases_display,
            test_report_display,
        ],
        js="() => [window.__target_delete_session_id || '']",
    )

    async def on_app_load():
        """页面首次载入或刷新时动态同步最新会话与消息及工作区与产物面板"""
        sessions = list_sessions()
        if sessions:
            sids = [s["session_id"] for s in sessions]
            if app_state.current_thread_id not in sids:
                app_state.current_thread_id = sessions[0]["session_id"]
        else:
            app_state.current_thread_id = create_session(
                title="新会话", workspace_path=app_state.current_workspace or ""
            )

        msgs = await get_session_messages_for_chatbot(app_state.current_thread_id)
        sess = get_session(app_state.current_thread_id)
        ws_path = sess.get("workspace_path", "") if sess else ""
        ws_val, status_html = get_workspace_status_for_session(ws_path)
        if ws_val and os.path.exists(ws_val):
            app_state.current_workspace = ws_val
        else:
            app_state.current_workspace = None

        d_html, tc_md, tr_md = await load_workflow_panels_for_session(app_state.current_thread_id)

        return (
            render_session_list_html(app_state.current_thread_id),
            msgs,
            msgs,
            ws_val,
            status_html,
            d_html,
            tc_md,
            tr_md,
        )

    demo.load(
        fn=on_app_load,
        inputs=[],
        outputs=[
            session_list_html,
            chat.chatbot,
            chat.chatbot_state,
            workspace_input,
            status_display,
            design_preview_display,
            test_cases_display,
            test_report_display,
        ],
    )

if __name__ == "__main__":
    import signal

    def _handle_exit(*args):
        print("\n[LC Agent] 服务已终止，正在退出...")
        os._exit(0)

    # 注册系统中断信号，防止 Windows 下 Gradio/Uvicorn/aiosqlite 后台线程挂起导致 Ctrl+C 卡死
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        demo.launch(theme=custom_theme, css=custom_css, head=fullscreen_head_html)
    finally:
        os._exit(0)
