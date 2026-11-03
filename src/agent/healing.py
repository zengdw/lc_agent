from langchain_core.messages import ToolMessage


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
