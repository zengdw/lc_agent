import os, asyncio
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from agent.session_manager import get_checkpointer
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from prompt.index import CODE_AGENT_SYSTEM_PROMPT
from mcp_tools.index import get_mcp_tools
from skills.index import list_available_skills, load_skill, get_skill_file_path
from tools.index import run_shell_command, get_current_time
from tools.rag_tools import retrieve_code_context

chat_model = init_chat_model(os.environ["OPENAI_MODEL"], model_provider="openai")


def prune_history_tool_messages(state: dict) -> List[BaseMessage]:
    """
    动态消息预处理：
    1. 确保开头包含 SystemMessage
    2. 严格按大模型协议规范（OpenAI 规范）保证：任何带 tool_calls 的 AIMessage
       紧随其后必须是对应的 ToolMessage，修复因中断重试导致的人类消息夹塞
    3. 将历史前几轮的超长 ToolMessage 压缩为简短占位符
    """
    raw_messages = list(state.get("messages", []))

    # 1. 如果开头没有 SystemMessage，则插入到最前面
    if not raw_messages or not isinstance(raw_messages[0], SystemMessage):
        raw_messages.insert(0, SystemMessage(content=CODE_AGENT_SYSTEM_PROMPT))

    # 2. 收集所有 ToolMessage 以便按 tool_call_id 精确重排
    tool_msgs_by_id = {}
    for m in raw_messages:
        if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                tool_msgs_by_id[tcid] = m

    # 规范化消息流：遇到带 tool_calls 的 AIMessage，立即紧随其后排入对应 ToolMessage
    ordered_messages: List[BaseMessage] = []
    used_tool_call_ids = set()

    for m in raw_messages:
        if isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool":
            tcid = getattr(m, "tool_call_id", None)
            # 若已在对应 AIMessage 之后排入过，跳过避免重复
            if tcid in used_tool_call_ids:
                continue
            ordered_messages.append(m)
        elif getattr(m, "tool_calls", None):
            ordered_messages.append(m)
            for tc in m.tool_calls:
                tcid = tc.get("id")
                if tcid:
                    if tcid in tool_msgs_by_id:
                        ordered_messages.append(tool_msgs_by_id[tcid])
                    else:
                        # 缺失占位 ToolMessage
                        ordered_messages.append(
                            ToolMessage(
                                content="[执行被用户中断或进程超时，未产生有效输出]",
                                tool_call_id=tcid,
                                name=tc.get("name", "tool"),
                            )
                        )
                    used_tool_call_ids.add(tcid)
        else:
            ordered_messages.append(m)

    # 3. 找到最后一个用户提问的位置（标识当前最新一轮对话起点）
    last_user_idx = -1
    for idx in range(len(ordered_messages) - 1, -1, -1):
        if getattr(ordered_messages[idx], "type", "") in ("user", "human"):
            last_user_idx = idx
            break

    # 4. 遍历并压缩最新一轮之前的历史 ToolMessage
    pruned_messages: List[BaseMessage] = []
    for idx, msg in enumerate(ordered_messages):
        if idx < last_user_idx and isinstance(msg, ToolMessage):
            tool_name = getattr(msg, "name", "tool") or "tool"
            raw_len = len(str(msg.content))
            pruned_content = (
                f"[历史工具 `{tool_name}` 输出已折叠，长度: {raw_len} 字符]"
            )

            pruned_msg = ToolMessage(
                content=pruned_content,
                tool_call_id=msg.tool_call_id,
                name=msg.name,
                status=getattr(msg, "status", "success"),
            )
            pruned_messages.append(pruned_msg)
        else:
            pruned_messages.append(msg)

    return pruned_messages


async def get_agent(workspace_path: str):
    if not workspace_path or not os.path.exists(workspace_path):
        raise ValueError(f"必须指定有效的项目工作区路径！当前传入: {workspace_path}")

    ws = os.path.abspath(workspace_path)
    tools = [
        retrieve_code_context,
        list_available_skills,
        load_skill,
        run_shell_command,
        get_current_time,
    ] + await get_mcp_tools(ws)

    agent = create_react_agent(
        model=chat_model,
        tools=tools,
        prompt=prune_history_tool_messages,
        checkpointer=await get_checkpointer(),
    )
    return agent


async def heal_agent_session(agent, config: dict) -> None:
    """自动检测并修复历史记录中因意外中断/超时导致未闭合的孤立 tool_calls"""
    try:
        current_state = await agent.aget_state(config)
        if not (current_state and current_state.values):
            return

        history_msgs = current_state.values.get("messages", [])
        existing_tool_ids = {
            getattr(m, "tool_call_id", None)
            for m in history_msgs
            if getattr(m, "type", "") == "tool" or isinstance(m, ToolMessage)
        }

        orphan_repair_messages = []
        for m in history_msgs:
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    tc_id = tc.get("id")
                    if tc_id and tc_id not in existing_tool_ids:
                        orphan_repair_messages.append(
                            ToolMessage(
                                content="[执行被用户中断或进程超时，未产生有效输出]",
                                tool_call_id=tc_id,
                                name=tc.get("name", "tool"),
                            )
                        )
                        existing_tool_ids.add(tc_id)

        if orphan_repair_messages:
            print(f"[Self-Healing] 自动修复 {len(orphan_repair_messages)} 个未闭合的工具调用")
            await agent.aupdate_state(config, {"messages": orphan_repair_messages})
    except Exception as e:
        print(f"[Self-Healing Warning] 状态自愈跳过: {e}")


async def main():
    agent = await get_agent()
    async for chunk, metadata in agent.astream(
        input={
            "messages": [
                {
                    "role": "user",
                    "content": "今天是几月几号",
                }
            ]
        },
        config={"configurable": {"thread_id": "1"}},
        stream_mode="messages",
    ):
        if isinstance(chunk, BaseMessage) and chunk.content:
            print(chunk.content, end="", flush=True)
        elif isinstance(chunk, str):
            print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
