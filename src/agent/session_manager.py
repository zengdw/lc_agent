import html
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiosqlite
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 确定数据库存储路径：统一存放在项目根目录下的 data/db/conversations.db
DATA_DIR = os.environ["DB_PATH"]
DB_PATH = os.path.join(DATA_DIR, "conversations.db")

_db_conn: Optional[sqlite3.Connection] = None
_async_conn: Optional[aiosqlite.Connection] = None
_async_checkpointer: Optional[AsyncSqliteSaver] = None


def get_db_connection() -> sqlite3.Connection:
    """获取或初始化全局 SQLite 数据库连接并开启 WAL 模式"""
    global _db_conn
    if _db_conn is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
        _db_conn.execute("PRAGMA journal_mode=WAL;")
        _init_db(_db_conn)
    return _db_conn


def _init_db(conn: sqlite3.Connection) -> None:
    """初始化会话元数据表"""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                workspace_path TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


async def get_checkpointer() -> AsyncSqliteSaver:
    """获取单例持久化 AsyncSqliteSaver 实例供 LangGraph 异步 Agent 使用"""
    global _async_checkpointer, _async_conn
    if _async_checkpointer is None:
        os.makedirs(DATA_DIR, exist_ok=True)
        get_db_connection()  # 确保 sessions 表初始化
        _async_conn = await aiosqlite.connect(DB_PATH)
        _async_checkpointer = AsyncSqliteSaver(_async_conn)
        await _async_checkpointer.setup()
    return _async_checkpointer


def create_session(
    title: str = "新会话", workspace_path: str = "", session_id: Optional[str] = None
) -> str:
    """创建新会话记录并返回 session_id"""
    sid = session_id or str(uuid.uuid4())
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO sessions (session_id, title, workspace_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?);
            """,
            (sid, title, workspace_path, now, now),
        )
    return sid


def list_sessions() -> List[Dict[str, Any]]:
    """按最近活跃时间倒序查询所有会话"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT session_id, title, workspace_path, created_at, updated_at
        FROM sessions
        ORDER BY updated_at DESC;
        """
    )
    rows = cursor.fetchall()
    return [
        {
            "session_id": row["session_id"],
            "title": row["title"],
            "workspace_path": row["workspace_path"],
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }
        for row in rows
    ]


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """获取单个会话的元数据"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, title, workspace_path, created_at, updated_at FROM sessions WHERE session_id = ?;",
        (session_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "session_id": row["session_id"],
        "title": row["title"],
        "workspace_path": row["workspace_path"],
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
    }


def update_session_title(session_id: str, title: str) -> None:
    """更新会话标题并更新修改时间"""
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?;",
            (title, now, session_id),
        )


def update_session_workspace(session_id: str, workspace_path: str) -> None:
    """更新会话绑定的工作区目录路径"""
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            "UPDATE sessions SET workspace_path = ?, updated_at = ? WHERE session_id = ?;",
            (workspace_path, now, session_id),
        )


def touch_session(session_id: str) -> None:
    """刷新会话的更新时间"""
    conn = get_db_connection()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with conn:
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?;",
            (now, session_id),
        )


def delete_session(session_id: str) -> None:
    """删除会话及其关联的 LangGraph 检查点记录"""
    conn = get_db_connection()
    with conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?;", (session_id,))
        # 级联清理 LangGraph 内部的检查点数据
        for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes", "writes"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?;", (session_id,))
            except sqlite3.OperationalError:
                pass
    conn.commit()



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


async def get_session_messages_for_chatbot(session_id: str) -> List[Dict[str, str]]:
    """从持久化 Checkpointer 读取历史消息并转换为 Gradio Chatbot 的标准展示格式"""
    checkpointer = await get_checkpointer()
    config = {"configurable": {"thread_id": session_id, "checkpoint_ns": ""}}
    state_tuple = await checkpointer.aget_tuple(config)
    if not state_tuple or not state_tuple.checkpoint:
        return []

    channel_values = state_tuple.checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])

    chatbot_messages: List[Dict[str, str]] = []
    current_ai_text = ""
    tool_calls_map: Dict[str, Dict[str, str]] = {}

    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage) or getattr(msg, "type", "") in (
            "human",
            "user",
        ):
            # 如果上一个 AI 消息尚未压入，先压入
            if current_ai_text:
                chatbot_messages.append(
                    {"role": "assistant", "content": current_ai_text}
                )
                current_ai_text = ""
                tool_calls_map.clear()
            user_text = (
                msg.content if isinstance(msg.content, str) else str(msg.content)
            )
            chatbot_messages.append({"role": "user", "content": user_text})

        elif isinstance(msg, AIMessage) or getattr(msg, "type", "") in (
            "ai",
            "assistant",
        ):
            # 捕获模型发起的工具调用
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_calls_map[tc_id] = {
                            "name": tc.get("name", "tool"),
                            "args": str(tc.get("args", "{}")),
                        }
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if text:
                current_ai_text += text

        elif isinstance(msg, ToolMessage) or getattr(msg, "type", "") == "tool":
            tc_id = getattr(msg, "tool_call_id", "")
            tc_info = tool_calls_map.get(tc_id, {})
            name = getattr(msg, "name", "") or tc_info.get("name", "tool")
            args = tc_info.get("args", "{}")
            output = str(msg.content)
            tool_html = format_tool_display_html(name, args, output)
            current_ai_text += tool_html

    if current_ai_text:
        chatbot_messages.append({"role": "assistant", "content": current_ai_text})

    return chatbot_messages


def get_session_choices() -> List[tuple[str, str]]:
    """获取格式化后的会话选择列表，格式为 [(显示标签, 会话ID), ...]"""
    sessions = list_sessions()
    choices = []
    for s in sessions:
        title = s.get("title") or "新会话"
        short_title = title if len(title) <= 22 else title[:21] + "..."
        updated = s.get("updated_at", "")
        time_label = updated[5:16] if len(updated) >= 16 else ""
        label = f"{short_title}\n{time_label}" if time_label else short_title
        choices.append((label, s["session_id"]))
    return choices


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


