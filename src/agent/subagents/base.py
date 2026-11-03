"""
子 Agent 通用运行器

子 Agent 特点：
- 一次性执行：每次委派都基于独立的消息上下文运行，执行结束即销毁；
- 不持久化：compile(checkpointer=False)，不会把内部过程写入用户会话的检查点；
- 对用户不可见：模型调用打上 TAG_NOSTREAM，其 token 不会流式输出到聊天界面，
  仅通过 on_progress 回调向主流程汇报“正在做什么”的精简进度。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import TAG_NOSTREAM
from langgraph.errors import GraphRecursionError

from agent.graph_builder import build_tool_agent_graph
from agent.memory import _content_to_text
from agent.model_config import create_chat_model
from prompt.common import build_workspace_context

SUBAGENT_RECURSION_LIMIT = int(os.environ.get("SUBAGENT_RECURSION_LIMIT", "200"))

TOOL_ACTION_LABELS = {
    "write_file": "写入文件",
    "edit_file": "编辑文件",
    "read_file": "读取文件",
    "read_text_file": "读取文件",
    "read_media_file": "读取媒体文件",
    "read_multiple_files": "批量读取文件",
    "create_directory": "创建目录",
    "list_directory": "浏览目录",
    "list_directory_with_sizes": "浏览目录",
    "directory_tree": "查看目录树",
    "move_file": "移动文件",
    "search_files": "搜索文件",
    "get_file_info": "查看文件信息",
    "run_shell_command": "执行命令",
    "retrieve_code_context": "检索代码库",
    "list_available_skills": "查看技能列表",
    "load_skill": "加载技能",
    "get_current_time": "获取时间",
}

ProgressCallback = Callable[[str], None]


@dataclass
class SubAgentResult:
    final_text: str = ""
    steps: int = 0
    actions: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.final_text) and not self.error


def describe_tool_call(name: str, args: dict) -> str:
    """把工具调用转换为简短的人类可读进度描述"""
    label = TOOL_ACTION_LABELS.get(name)
    if not label:
        label = "联网检索" if "tavily" in name.lower() or "search" in name.lower() else f"调用 {name}"
    detail = ""
    if isinstance(args, dict):
        if args.get("path"):
            detail = Path(str(args["path"])).name or str(args["path"])
        elif args.get("paths") and isinstance(args["paths"], list):
            detail = "、".join(Path(str(p)).name for p in args["paths"][:3])
        elif args.get("command"):
            detail = str(args["command"]).strip().splitlines()[0]
        elif args.get("query"):
            detail = str(args["query"])
        elif args.get("skill_name"):
            detail = str(args["skill_name"])
    if len(detail) > 60:
        detail = detail[:57] + "..."
    return f"{label} {detail}".strip()


async def run_subagent(
    *,
    role: str,
    system_prompt: str,
    task: str,
    tools: Sequence,
    workspace: str,
    on_progress: Optional[ProgressCallback] = None,
    recursion_limit: Optional[int] = None,
) -> SubAgentResult:
    """运行一个一次性子 Agent，返回其最终回复与执行统计"""
    result = SubAgentResult()

    def _progress(text: str) -> None:
        result.actions.append(text)
        if on_progress:
            try:
                on_progress(text)
            except Exception:
                pass

    try:
        model = create_chat_model(role)
    except Exception as e:
        result.error = f"模型初始化失败: {e}"
        return result

    full_prompt = system_prompt + build_workspace_context(workspace)
    graph = build_tool_agent_graph(
        model=model,
        tools=tools,
        prompt_fn=lambda _state: full_prompt,
        checkpointer=False,
        name=f"{role}_agent",
        run_config={"tags": [TAG_NOSTREAM], "run_name": f"{role}_agent"},
    )

    try:
        async for update in graph.astream(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": recursion_limit or SUBAGENT_RECURSION_LIMIT},
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue
            agent_update = update.get("agent")
            if not isinstance(agent_update, dict):
                continue
            for msg in agent_update.get("messages", []) or []:
                if not isinstance(msg, AIMessage):
                    continue
                result.steps += 1
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        _progress(describe_tool_call(tc.get("name", ""), tc.get("args") or {}))
                else:
                    text = _content_to_text(msg.content).strip()
                    if text:
                        result.final_text = text
    except GraphRecursionError:
        result.error = f"执行步数超过上限（{recursion_limit or SUBAGENT_RECURSION_LIMIT}），任务被中止，可能只完成了部分工作。"
    except Exception as e:
        import traceback

        traceback.print_exc()
        result.error = f"执行异常: {e}"

    print(f"[SubAgent:{role}] 完成，步数={result.steps}，动作数={len(result.actions)}，错误={result.error or '无'}")
    return result
