"""聊天消息中的工具调用 / 阶段卡片 / 实时进度渲染（实时流与历史回放共用）"""

import html
from typing import List, Optional

from agent.delegation import DELEGATE_CODING, DELEGATE_DESIGN, DELEGATE_TESTING

# 委派工具在界面上以“工作阶段”呈现，不暴露内部子 Agent
STAGE_TOOL_META = {
    DELEGATE_DESIGN: {"icon": "🎨", "title": "方案设计 & 测试用例", "cls": "design"},
    DELEGATE_CODING: {"icon": "💻", "title": "编码实现", "cls": "coding"},
    DELEGATE_TESTING: {"icon": "🧪", "title": "测试验证", "cls": "testing"},
}

LIVE_STAGE_META = {
    "design": ("🎨", "方案设计中"),
    "test_cases": ("📋", "正在生成测试用例"),
    "coding": ("💻", "编码实现中"),
    "testing": ("🧪", "测试验证中"),
}


def format_tool_display_html(tool_name: str, tool_args: str, tool_output: str) -> str:
    """将工具调用及输出格式化为美观的高亮折叠卡片 HTML"""
    escaped_name = html.escape(str(tool_name))
    escaped_args = html.escape(str(tool_args))
    escaped_output = html.escape(str(tool_output))
    return (
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


def _stage_badge(tool_name: str, output: str) -> tuple[str, str]:
    text = (output or "").strip()
    first_line = text.split("\n", 1)[0] if text else ""

    if tool_name == DELEGATE_TESTING:
        # 优先解析测试明确结论，严防统计文本中的“失败 0”被误判为失败
        if "✅" in first_line or ("通过" in first_line and "❌" not in first_line and "未通过" not in first_line):
            return "通过", "badge-success"
        if "❌" in first_line or "未通过" in first_line:
            return "未通过", "badge-failed"
        if "⚠️" in first_line:
            return "待确认", "badge-warn"
        return "已完成", "badge-warn"

    if tool_name == DELEGATE_DESIGN:
        if "设计失败" in first_line or "无法开始" in first_line:
            return "失败", "badge-failed"
        return "待确认", "badge-warn"

    if tool_name == DELEGATE_CODING:
        if "失败】" in first_line or "无法开始" in first_line or "执行失败" in first_line:
            return "失败", "badge-failed"
        return "已完成", "badge-success"

    # 通用兜底匹配（精准匹配错误标志，避免普通文本误伤）
    if "失败】" in first_line or "无法开始" in first_line or "执行失败" in first_line:
        return "失败", "badge-failed"
    return "已完成", "badge-success"


def format_stage_card_html(tool_name: str, tool_output: str) -> str:
    """委派工具 → 工作阶段卡片（默认折叠，展开可查看阶段产出摘要）"""
    meta = STAGE_TOOL_META[tool_name]
    badge, badge_cls = _stage_badge(tool_name, tool_output)
    # 去掉仅面向主 Agent 的“【下一步】”指令
    body = (tool_output or "").split("【下一步】", 1)[0].strip()
    return (
        f'\n\n<details class="stage-card stage-{meta["cls"]}">\n'
        f'<summary class="stage-card-summary"><span class="stage-card-icon">{meta["icon"]}</span>'
        f'<strong>{meta["title"]}</strong>'
        f'<span class="stage-badge {badge_cls}">{badge}</span></summary>\n'
        f'<div class="stage-card-body">{html.escape(body)}</div>\n'
        f"</details>\n\n"
    )


def render_tool_card(tool_name: str, tool_args: str, tool_output: str) -> str:
    if tool_name in STAGE_TOOL_META:
        return format_stage_card_html(tool_name, tool_output)
    return format_tool_display_html(tool_name, tool_args, tool_output)


def render_live_progress(stage: Optional[str], actions: List[str], pending_tools: List[str]) -> str:
    """流式过程中附加在回复末尾的实时进度块（不持久化）"""
    if stage and stage in LIVE_STAGE_META:
        icon, title = LIVE_STAGE_META[stage]
        items = "".join(
            f'<li>{html.escape(a)}</li>' for a in actions[-5:]
        ) or "<li>准备中…</li>"
        return (
            f'\n\n<div class="live-progress live-{stage}">'
            f'<div class="live-progress-head"><span class="live-spinner"></span>{icon} <strong>{title}</strong>'
            f'<span class="live-count">已执行 {len(actions)} 步</span></div>'
            f'<ul class="live-progress-list">{items}</ul></div>'
        )
    if pending_tools:
        names = "、".join(html.escape(n) for n in pending_tools)
        return (
            f'\n\n<div class="live-progress"><div class="live-progress-head">'
            f'<span class="live-spinner"></span>⚙️ 正在执行工具：<code>{names}</code></div></div>'
        )
    return ""
