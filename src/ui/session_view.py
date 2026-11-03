"""会话视图：历史消息回放、会话列表渲染与会话工作流状态读取"""

import html
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.memory import _content_to_text
from agent.session_manager import get_checkpointer, list_sessions
from ui.render import render_tool_card


async def _get_channel_values(session_id: str) -> dict:
    checkpointer = await get_checkpointer()
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
    state_tuple = await checkpointer.aget_tuple(config)
    if not state_tuple or not state_tuple.checkpoint:
        return {}
    return state_tuple.checkpoint.get("channel_values", {}) or {}


async def get_session_workflow(session_id: str) -> dict:
    """读取会话持久化的多 Agent 工作流进度"""
    values = await _get_channel_values(session_id)
    return values.get("workflow") or {}


async def get_session_messages_for_chatbot(session_id: str) -> List[Dict[str, str]]:
    """从持久化 Checkpointer 读取历史消息并转换为 Gradio Chatbot 的标准展示格式"""
    messages = (await _get_channel_values(session_id)).get("messages", [])

    chatbot_messages: List[Dict[str, str]] = []
    current_ai_text = ""
    tool_calls_map: Dict[str, Dict[str, str]] = {}

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") in ("human", "user"):
            # 如果上一个 AI 消息尚未压入，先压入
            if current_ai_text:
                chatbot_messages.append({"role": "assistant", "content": current_ai_text})
                current_ai_text = ""
                tool_calls_map.clear()
            chatbot_messages.append({"role": "user", "content": _content_to_text(msg.content)})

        elif isinstance(msg, AIMessage) or getattr(msg, "type", "") in ("ai", "assistant"):
            # 捕获模型发起的工具调用
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id")
                if tc_id:
                    tool_calls_map[tc_id] = {
                        "name": tc.get("name", "tool"),
                        "args": str(tc.get("args", "{}")),
                    }
            text = _content_to_text(msg.content)
            if text:
                current_ai_text += text

        elif isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
            tc_info = tool_calls_map.get(getattr(msg, "tool_call_id", ""), {})
            name = getattr(msg, "name", "") or tc_info.get("name", "tool")
            current_ai_text += render_tool_card(name, tc_info.get("args", "{}"), str(msg.content))

    if current_ai_text:
        chatbot_messages.append({"role": "assistant", "content": current_ai_text})

    return chatbot_messages


def render_session_list_html(active_session_id: str) -> str:
    """生成完全受控的高颜值自定义历史会话列表卡片 HTML"""
    sessions = list_sessions()
    if not sessions:
        return (
            '<div class="session-list-empty">'
            '<span class="empty-icon">💬</span>'
            '<div>暂无历史会话</div>'
            '<small>点击上方「➕ 新建会话」开启对话</small>'
            '</div>'
        )

    trash_svg = (
        '<svg viewBox="0 0 24 24" width="13" height="13" stroke="currentColor" '
        'stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="3 6 5 6 21 6"></polyline>'
        '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>'
        '</svg>'
    )

    cards_html = []
    for s in sessions:
        sid = s["session_id"]
        is_active = (sid == active_session_id)
        active_class = " selected" if is_active else ""

        raw_title = s.get("title") or "新会话"
        title_escaped = html.escape(raw_title)

        updated = s.get("updated_at", "")
        time_label = updated[5:16] if len(updated) >= 16 else ""
        time_escaped = html.escape(time_label)

        time_div = f'<div class="session-time">{time_escaped}</div>' if time_escaped else ""

        card = (
            f'<div class="session-card{active_class}" data-sid="{sid}" '
            f'onclick="window.handleSessionClick(event, \'{sid}\')" role="button" tabindex="0">'
            f'  <div class="session-radio-indicator">'
            f'    <span class="session-radio-dot"></span>'
            f'  </div>'
            f'  <div class="session-info">'
            f'    <div class="session-title" title="{title_escaped}">{title_escaped}</div>'
            f'    {time_div}'
            f'  </div>'
            f'  <button type="button" class="session-del-btn" title="删除此会话" '
            f'    onclick="window.handleSessionDelete(event, \'{sid}\')" aria-label="删除此会话">'
            f'    {trash_svg}'
            f'  </button>'
            f'</div>'
        )
        cards_html.append(card)

    return f'<div class="custom-session-list">{"".join(cards_html)}</div>'
