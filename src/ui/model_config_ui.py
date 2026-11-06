"""
各个 Agent 的独立模型配置面板（居中模态弹窗形式）
支持在界面上为 主 Agent、设计 Agent、编码 Agent、测试 Agent 独立配置不同的模型：
Provider、Model Name、Base URL、API Key、Temperature。
修改后保存到 data/agent_models.json 并热更新全局 Agent 缓存。
"""

import gradio as gr
from agent.model_config import (
    AGENT_ROLES,
    CONFIG_FIELDS,
    get_override,
    get_model_config,
    get_inherited_config,
    save_override,
    reset_override,
)
from ui.state import app_state


def build_model_config_ui():
    """构建多 Agent 模型配置弹窗模态框"""
    with gr.Column(visible=False, elem_id="model-config-modal-overlay") as modal_wrapper:
        with gr.Column(elem_id="model-config-modal-container"):
            # 顶部 Header：标题区 + 优雅关闭按钮
            with gr.Row(elem_classes=["modal-header-row"]):
                gr.HTML(
                    '<div class="modal-title-box">'
                    '  <div class="modal-icon-badge">⚙️</div>'
                    '  <div>'
                    '    <h3 class="modal-main-title">多 Agent 独立模型配置</h3>'
                    '    <p class="modal-sub-title">为各个 Agent 分配专属模型。留空将自动继承 <code>.env</code> 全局默认配置。</p>'
                    '  </div>'
                    '</div>'
                )
                close_modal_btn = gr.Button("✕", variant="secondary", elem_id="close-modal-btn", scale=0)

            components_by_role = {}

            with gr.Tabs(elem_classes=["modal-tabs"]):
                for role, info in AGENT_ROLES.items():
                    with gr.TabItem(f"{info['label']}"):
                        # 角色职责信息提示栏
                        gr.HTML(
                            f'<div class="role-desc-banner">'
                            f'  <strong>角色职责</strong>：{info["desc"]}'
                            f'  <div class="role-env-tag">环境变量：<code>{info["env_prefix"]}_MODEL / _PROVIDER / _BASE_URL / _API_KEY</code></div>'
                            f'</div>'
                        )
                        
                        cur_override = get_override(role)
                        cur_effective = get_model_config(role)
                        inherited = get_inherited_config(role)

                        with gr.Column(elem_classes=["modal-form-body"]):
                            # 第 1 行：Provider + Model Name + Temperature
                            with gr.Row(elem_classes=["modal-form-row"]):
                                provider_in = gr.Textbox(
                                    label="Provider (默认 openai)",
                                    value=cur_override.get("provider", ""),
                                    placeholder=inherited.get("provider", "openai"),
                                    scale=3,
                                )
                                model_in = gr.Textbox(
                                    label="模型名称 (Model)",
                                    value=cur_override.get("model", ""),
                                    placeholder=f"继承: {inherited.get('model', '(未配置)')}",
                                    scale=5,
                                )
                                temp_in = gr.Textbox(
                                    label="采样温度 (Temperature)",
                                    value=cur_override.get("temperature", ""),
                                    placeholder=inherited.get("temperature", "默认"),
                                    scale=4,
                                )

                            # 第 2 行：Base URL + API Key
                            with gr.Row(elem_classes=["modal-form-row"]):
                                base_url_in = gr.Textbox(
                                    label="API Base URL",
                                    value=cur_override.get("base_url", ""),
                                    placeholder=inherited.get("base_url", "https://api.openai.com/v1"),
                                    scale=6,
                                )
                                api_key_in = gr.Textbox(
                                    label="API Key",
                                    value=cur_override.get("api_key", ""),
                                    placeholder="继承默认 API Key" if inherited.get("api_key") else "请输入 API Key",
                                    type="password",
                                    scale=6,
                                )

                            # 当前生效模型状态栏
                            with gr.Row(elem_classes=["modal-status-row"]):
                                role_status = gr.HTML(
                                    value=f'<div class="effective-status-badge">⚡ 当前生效模型：<code>{cur_effective["provider"]}:{cur_effective["model"] or "未配置"}</code></div>'
                                )

                            # 底部操作按钮栏
                            with gr.Row(elem_classes=["modal-action-row"]):
                                save_btn = gr.Button("💾 保存该 Agent 配置", variant="primary", scale=3, elem_classes=["modal-save-btn"])
                                reset_btn = gr.Button("🔄 恢复继承默认", variant="secondary", scale=2, elem_classes=["modal-reset-btn"])

                            components_by_role[role] = {
                                "provider": provider_in,
                                "model": model_in,
                                "temperature": temp_in,
                                "base_url": base_url_in,
                                "api_key": api_key_in,
                                "status": role_status,
                                "save_btn": save_btn,
                                "reset_btn": reset_btn,
                            }

            # 绑定各角色的保存与重置操作
            for r, comps in components_by_role.items():
                def make_save_handler(target_role):
                    def _save(p, m, t, bu, ak):
                        try:
                            save_override(target_role, {
                                "provider": p,
                                "model": m,
                                "temperature": t,
                                "base_url": bu,
                                "api_key": ak,
                            })
                            app_state.invalidate_agent()
                            eff = get_model_config(target_role)
                            return f'<div class="effective-status-badge status-ok">✅ 保存成功！当前生效: <code>{eff["provider"]}:{eff["model"]}</code></div>'
                        except Exception as e:
                            return f'<div class="effective-status-badge status-err">❌ 保存失败: {e}</div>'
                    return _save

                comps["save_btn"].click(
                    fn=make_save_handler(r),
                    inputs=[comps["provider"], comps["model"], comps["temperature"], comps["base_url"], comps["api_key"]],
                    outputs=[comps["status"]],
                )

                def make_reset_handler(target_role, r_comps):
                    def _reset():
                        reset_override(target_role)
                        app_state.invalidate_agent()
                        eff = get_model_config(target_role)
                        msg = f'<div class="effective-status-badge status-reset">🔄 已恢复继承。当前生效: <code>{eff["provider"]}:{eff["model"]}</code></div>'
                        return "", "", "", "", "", msg
                    return _reset

                comps["reset_btn"].click(
                    fn=make_reset_handler(r, comps),
                    inputs=[],
                    outputs=[
                        comps["provider"],
                        comps["model"],
                        comps["temperature"],
                        comps["base_url"],
                        comps["api_key"],
                        comps["status"],
                    ],
                )

            # 关闭弹窗按钮
            close_modal_btn.click(
                fn=lambda: gr.update(visible=False),
                inputs=[],
                outputs=[modal_wrapper],
            )

    return modal_wrapper
