"""
通用「模型 ↔ 工具」循环图构建器，主 Agent 与各子 Agent 共用

图结构：START → memory → agent ─(有工具调用)→ tools → memory → agent ...
                                └(无工具调用)→ END
memory 节点在每次调用模型前检查 token 长度，必要时压缩滑动窗口外的历史
"""

from functools import lru_cache
from typing import Callable, Optional, Sequence, Type

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.memory import AgentState, build_model_input, get_summary_model, make_memory_node


@lru_cache(maxsize=1)
def get_shared_summary_model():
    """历史摘要压缩使用独立模型（SUMMARY_* 环境变量，未配置则回退到主模型配置）"""
    return get_summary_model()


def build_tool_agent_graph(
    *,
    model,
    tools: Sequence,
    prompt_fn: Callable[[dict], str],
    state_schema: Type[AgentState] = AgentState,
    checkpointer=None,
    name: str = "agent",
    run_config: Optional[dict] = None,
):
    """
    :param model: 尚未绑定工具的聊天模型（可带 with_config 等配置）
    :param tools: 可调用的工具列表
    :param prompt_fn: 根据当前 state 生成系统提示词的函数（便于注入动态上下文）
    :param checkpointer: 持久化器；子 Agent 传 False 表示不持久化、也不继承父图的持久化器
    :param run_config: 绑定工具后附加的运行配置（如 tags / run_name）
    """
    model_with_tools = model.bind_tools(list(tools))
    if run_config:
        model_with_tools = model_with_tools.with_config(**run_config)

    async def call_model(state: dict) -> dict:
        """agent 节点：注入系统提示词 + 历史摘要，规范化/折叠消息后调用模型"""
        model_input = build_model_input(state, prompt_fn(state))
        response = await model_with_tools.ainvoke(model_input)
        return {"messages": [response]}

    graph = StateGraph(state_schema)
    graph.add_node("memory", make_memory_node(get_shared_summary_model()))
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(list(tools)))

    graph.add_edge(START, "memory")
    graph.add_edge("memory", "agent")
    graph.add_conditional_edges("agent", tools_condition)  # → "tools" 或 END
    graph.add_edge("tools", "memory")

    return graph.compile(checkpointer=checkpointer, name=name)
