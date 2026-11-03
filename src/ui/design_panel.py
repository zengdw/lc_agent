"""
设计稿与测试结果面板渲染模块
用于在 Gradio 界面右侧实时或按会话渲染：
1. 设计稿预览（支持自包含 HTML 的 iframe 预览及图片预览）
2. 测试用例与测试结果报告（Markdown 渲染）
"""

import html
import base64
from pathlib import Path
from typing import Optional
from agent.workflow import (
    DESIGN_EXTS,
    IMAGE_EXTS,
    find_design_artifacts,
    read_text_file,
)
from ui.session_view import get_session_workflow


def render_design_preview_html(artifacts: list[str]) -> str:
    """将设计稿列表转换为 iframe / img 预览 HTML"""
    if not artifacts:
        return (
            '<div class="design-preview-empty">'
            '<div class="empty-icon">🎨</div>'
            '<div>暂无设计稿产物</div>'
            '<small>当主 Agent 分配任务给设计 Agent 并在产出设计稿后，此处将自动呈现可视化原型供您确认。</small>'
            '</div>'
        )

    # 优先寻找 index.html 或第一个 html 文件
    html_file: Optional[Path] = None
    images: list[Path] = []
    
    for a in artifacts:
        p = Path(a)
        if not p.exists():
            continue
        if p.suffix.lower() in (".html", ".htm") and html_file is None:
            html_file = p
        elif p.suffix.lower() in IMAGE_EXTS:
            images.append(p)

    parts = []

    # 1. 渲染 HTML 设计原型
    if html_file and html_file.exists():
        try:
            content = html_file.read_text(encoding="utf-8", errors="replace")
            # 通过 srcdoc 嵌入沙箱 iframe 预览，防止主页面被样式污染
            escaped_srcdoc = html.escape(content, quote=True)
            parts.append(
                f'<div class="design-file-badge">📄 原型页面：<code>{html.escape(html_file.name)}</code></div>'
                f'<div class="design-iframe-container">'
                f'<iframe class="design-preview-iframe" sandbox="allow-scripts allow-forms" srcdoc="{escaped_srcdoc}"></iframe>'
                f'</div>'
            )
        except Exception as e:
            parts.append(f'<div class="design-error">读取设计文件失败: {html.escape(str(e))}</div>')

    # 2. 渲染补充的图片/SVG
    if images:
        parts.append('<div class="design-images-grid">')
        for img in images:
            try:
                if img.suffix.lower() == ".svg":
                    svg_content = img.read_text(encoding="utf-8", errors="replace")
                    parts.append(
                        f'<div class="design-image-card">'
                        f'<div class="design-img-title">🖼️ {html.escape(img.name)}</div>'
                        f'<div class="design-svg-box">{svg_content}</div>'
                        f'</div>'
                    )
                else:
                    b64 = base64.b64encode(img.read_bytes()).decode("ascii")
                    mime = f"image/{img.suffix.lower().lstrip('.')}"
                    parts.append(
                        f'<div class="design-image-card">'
                        f'<div class="design-img-title">🖼️ {html.escape(img.name)}</div>'
                        f'<img src="data:{mime};base64,{b64}" class="design-img-preview" />'
                        f'</div>'
                    )
            except Exception as e:
                parts.append(f'<div class="design-error">读取图片失败: {html.escape(str(e))}</div>')
        parts.append('</div>')

    return "\n".join(parts)


def get_workflow_panel_data(wf: dict) -> tuple[str, str, str]:
    """
    根据 workflow dict 返回：
    1. design_preview_html (设计预览 HTML)
    2. test_cases_md (测试用例 Markdown)
    3. test_report_md (测试报告 Markdown)
    """
    artifacts = wf.get("design_artifacts", []) or []
    design_html = render_design_preview_html(artifacts)

    test_cases_path = wf.get("test_cases_path", "")
    test_cases_md = read_text_file(test_cases_path) if test_cases_path else "*暂未生成测试用例（将在设计 Agent 产出设计稿后自动生成）*"

    test_report_path = wf.get("test_report_path", "")
    test_report_md = read_text_file(test_report_path) if test_report_path else "*暂无测试报告（将在编码 Agent 完成实现后由测试 Agent 执行测试并生成）*"

    return design_html, test_cases_md, test_report_md


async def load_workflow_panels_for_session(session_id: str) -> tuple[str, str, str]:
    """读取指定会话的 workflow 状态并返回面板展示数据"""
    wf = await get_session_workflow(session_id)
    return get_workflow_panel_data(wf)
