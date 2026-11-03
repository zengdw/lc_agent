"""界面层全局状态：当前会话、当前工作区与主 Agent 实例缓存"""

import asyncio
import os
from typing import List, Optional

from langgraph.graph.state import CompiledStateGraph

from agent.index import get_agent
from agent.model_config import config_signature
from agent.session_manager import create_session, list_sessions
from ui.session_view import get_session_messages_for_chatbot


class AppState:
    def __init__(self) -> None:
        self.agent: Optional[CompiledStateGraph] = None
        self.agent_key: Optional[tuple] = None
        self.current_workspace: Optional[str] = None
        self.current_thread_id: str = ""

    def bootstrap(self) -> List[dict]:
        """初始化会话：优先加载数据库中最近活跃会话，若无则自动新建；返回初始聊天记录"""
        existing = list_sessions()
        if existing:
            self.current_thread_id = existing[0]["session_id"]
            self.current_workspace = existing[0].get("workspace_path") or None
            return asyncio.run(get_session_messages_for_chatbot(self.current_thread_id))
        self.current_thread_id = create_session(title="新会话")
        return []

    async def get_cached_agent(self, workspace_path: str) -> CompiledStateGraph:
        """按 (工作区, 主 Agent 模型配置) 懒加载并缓存主 Agent 实例"""
        ws = os.path.abspath(workspace_path.strip())
        key = (ws, config_signature("main"))
        if self.agent is None or self.agent_key != key:
            print(f"[Workspace] 构建主 Agent: 工作区={ws}")
            self.current_workspace = ws
            self.agent = await get_agent(ws)
            self.agent_key = key
        return self.agent

    def invalidate_agent(self) -> None:
        """模型配置变化后丢弃缓存，下次对话时按新配置重建"""
        self.agent = None
        self.agent_key = None


app_state = AppState()
