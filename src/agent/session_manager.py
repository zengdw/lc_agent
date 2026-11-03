import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import aiosqlite
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
