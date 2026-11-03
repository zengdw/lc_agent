"""
主 Agent（调度者）：直接处理简单任务；复杂任务通过委派工具分派给设计 / 编码 / 测试子 Agent
"""

import os
from typing import Dict, List

from agent.delegation import make_delegation_tools
from agent.graph_builder import build_tool_agent_graph
from agent.model_config import create_chat_model
from agent.session_manager import get_checkpointer
from agent.workflow import OrchestratorState, describe_workflow_for_prompt
from mcp_tools.index import get_mcp_tools
from prompt.common import build_workspace_context
from prompt.main_prompt import MAIN_AGENT_SYSTEM_PROMPT
from skills.index import list_available_skills, load_skill
from tools.index import get_current_time, run_shell_command
from tools.rag_tools import retrieve_code_context

# MCP 工具按工作区缓存，避免切换模型配置重建 Agent 时重复拉起 npx 子进程
_mcp_tools_cache: Dict[str, List] = {}


async def _get_mcp_tools_cached(ws: str) -> List:
    if ws not in _mcp_tools_cache:
        _mcp_tools_cache[ws] = await get_mcp_tools(ws)
    return _mcp_tools_cache[ws]


async def get_worker_tools(ws: str) -> List:
    """主 Agent 与子 Agent 共用的基础工具集"""
    return [
        retrieve_code_context,
        list_available_skills,
        load_skill,
        run_shell_command,
        get_current_time,
    ] + await _get_mcp_tools_cached(ws)


async def get_agent(workspace_path: str):
    if not workspace_path or not os.path.exists(workspace_path):
        raise ValueError(f"必须指定有效的项目工作区路径！当前传入: {workspace_path}")

    ws = os.path.abspath(workspace_path)
    worker_tools = await get_worker_tools(ws)
    tools = worker_tools + make_delegation_tools(ws, worker_tools)
    workspace_context = build_workspace_context(ws)

    def prompt_fn(state: dict) -> str:
        return (
            MAIN_AGENT_SYSTEM_PROMPT
            + workspace_context
            + describe_workflow_for_prompt(state.get("workflow"))
        )

    return build_tool_agent_graph(
        model=create_chat_model("main"),
        tools=tools,
        prompt_fn=prompt_fn,
        state_schema=OrchestratorState,
        checkpointer=await get_checkpointer(),
        name="main_agent",
    )
