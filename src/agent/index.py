import os, asyncio
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from agent.session_manager import get_checkpointer
from agent.memory import AgentState, build_model_input, make_memory_node, get_summary_model
from langchain_core.messages import BaseMessage, ToolMessage
from prompt.index import CODE_AGENT_SYSTEM_PROMPT
from mcp_tools.index import get_mcp_tools
from skills.index import list_available_skills, load_skill, get_skill_file_path
from tools.index import run_shell_command, get_current_time
from tools.rag_tools import retrieve_code_context

chat_model = init_chat_model(os.environ["OPENAI_MODEL"], model_provider="openai")
# 历史摘要压缩使用独立模型（SUMMARY_* 环境变量，未配置则回退到主模型配置）
summary_model = get_summary_model()


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

    model_with_tools = chat_model.bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        """agent 节点：注入系统提示词 + 历史摘要，规范化/折叠消息后调用模型"""
        model_input = build_model_input(state, CODE_AGENT_SYSTEM_PROMPT)
        response = await model_with_tools.ainvoke(model_input)
        return {"messages": [response]}

    # 图结构：START → memory → agent ─(有工具调用)→ tools → memory → agent ...
    #                                  └(无工具调用)→ END
    # memory 节点在每次调用模型前检查 token 长度，必要时压缩滑动窗口外的历史
    graph = StateGraph(AgentState)
    graph.add_node("memory", make_memory_node(summary_model))
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "memory")
    graph.add_edge("memory", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" 或 END
    graph.add_edge("tools", "memory")

    agent = graph.compile(checkpointer=await get_checkpointer())
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
