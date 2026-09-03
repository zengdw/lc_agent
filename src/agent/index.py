import os, asyncio
from typing import List
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from prompt.index import CODE_AGENT_SYSTEM_PROMPT
from mcp_tools.index import get_mcp_tools
from skills.index import list_available_skills, load_skill, get_skill_file_path
from tools.index import run_shell_command, get_current_time
from tools.rag_tools import retrieve_code_context

chat_model = init_chat_model(os.environ["OPENAI_MODEL"])


def prune_history_tool_messages(state: dict) -> List[BaseMessage]:
    """
    动态消息预处理：
    1. 确保开头包含 SystemMessage
    2. 找到最新一轮用户提问的起点
    3. 将历史前几轮的超长 ToolMessage 压缩为简短占位符
    """
    messages = list(state.get("messages", []))

    # 1. 如果开头没有 SystemMessage，则插入到最前面
    if not messages or not isinstance(messages[0], SystemMessage):
        messages.insert(0, SystemMessage(content=CODE_AGENT_SYSTEM_PROMPT))

    # 2. 找到最后一个用户提问的位置（标识当前最新一轮对话起点）
    last_user_idx = -1
    for idx in range(len(messages) - 1, -1, -1):
        if getattr(messages[idx], "type", "") in ("user", "human"):
            last_user_idx = idx
            break

    # 3. 遍历并压缩最新一轮之前的历史 ToolMessage
    pruned_messages: List[BaseMessage] = []
    for idx, msg in enumerate(messages):
        # 仅对最新一轮之前的历史 ToolMessage 进行内容压缩，不影响 State 内的真实完整数据
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
            # 开头的 SystemMessage 以及其他正常消息会自然原样装入
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
        checkpointer=InMemorySaver(),
    )
    return agent


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
