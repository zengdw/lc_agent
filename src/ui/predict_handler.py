"""
Gradio 预测函数与流式消息生成逻辑
对接 LangGraph 主 Agent，处理工具调用、子 Agent 进度流推送、错误自愈与实时状态渲染
"""

import os
import html
import asyncio
from typing import AsyncGenerator
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.healing import heal_agent_session
from agent.session_manager import (
    create_session,
    get_session,
    update_session_title,
    update_session_workspace,
    touch_session,
)
from tools.rag_tools import auto_sync_codebase
from ui.state import app_state
from ui.render import render_tool_card, render_live_progress
from ui.design_panel import get_workflow_panel_data
from agent.workflow import empty_workflow


async def predict(
    message: str,
    history: list,
    workspace_path: str = "",
) -> AsyncGenerator[tuple[str, str, str, str], None]:
    """
    Gradio 预测函数：流式输出主 Agent 回复并同步更新右侧工作区与设计/测试面板
    Yield 格式：(ai_response_text, design_preview_html, test_cases_md, test_report_md)
    """
    # 默认面板保持原样（或空）
    cur_design_html = ""
    cur_test_cases_md = "*暂未生成测试用例*"
    cur_test_report_md = "*暂无测试报告*"

    def _pack(resp: str):
        return (resp, cur_design_html, cur_test_cases_md, cur_test_report_md)

    # 1. 严格检查工作区
    ws = (workspace_path or "").strip()
    if not ws or not os.path.exists(ws):
        yield _pack(
            "⚠️ **请先在上方控制台绑定项目工作区**\n\n"
            "点击「📂 **浏览选择文件夹**」选择目标本地代码库，智能体将自动建立 RAG 代码符号知识库后即可开始对话与多 Agent 协作！"
        )
        return

    clean_msg = message.strip()
    if clean_msg.lower() in ("/new", "/clear", "/reset"):
        app_state.current_thread_id = create_session(title="新会话", workspace_path=ws)
        yield _pack("🔄 **已创建全新会话！** 历史上下文已就绪，请输入新的需求开启交互。")
        return

    # 会话元数据维护
    sess = get_session(app_state.current_thread_id)
    if sess and sess.get("title") in ("新会话", ""):
        first_line = clean_msg.split("\n")[0].strip()
        summary_title = first_line if len(first_line) <= 18 else first_line[:17] + "..."
        update_session_title(app_state.current_thread_id, summary_title)
    else:
        touch_session(app_state.current_thread_id)

    if sess and sess.get("workspace_path") != ws:
        update_session_workspace(app_state.current_thread_id, ws)

    agent = await app_state.get_cached_agent(ws)

    config: RunnableConfig = {
        "configurable": {"thread_id": app_state.current_thread_id},
    }

    # 自动会话自愈
    await heal_agent_session(agent, config)

    response_text = ""
    current_tool_name = ""
    current_tool_args = ""
    active_stage: str | None = None
    subagent_actions: list[str] = []
    pending_tools: list[str] = []

    try:
        # stream_mode=["messages", "custom", "updates"]
        async for chunk_item in agent.astream(
            {"messages": [("user", message)]},
            config=config,
            stream_mode=["messages", "custom", "updates"],
        ):
            mode, payload = chunk_item[0], chunk_item[1]

            # 1. 处理自定义进度流（子 Agent 委派过程）
            if mode == "custom" and isinstance(payload, dict):
                src = payload.get("source")
                if src == "workflow":
                    event_type = payload.get("type")
                    if event_type == "stage":
                        st = payload.get("status")
                        if st == "start":
                            active_stage = payload.get("stage")
                            subagent_actions = []
                        elif st in ("end", "failed"):
                            active_stage = None
                    elif event_type == "progress":
                        t = payload.get("text")
                        if t:
                            subagent_actions.append(t)
                
                live_block = render_live_progress(active_stage, subagent_actions, pending_tools)
                yield _pack(response_text + live_block)

            # 2. 处理节点更新 updates：提取 workflow 并刷新右侧设计/测试面板
            elif mode == "updates" and isinstance(payload, dict):
                # 检查是否包含了 workflow 的更新
                for node_name, node_val in payload.items():
                    if isinstance(node_val, dict) and "workflow" in node_val:
                        wf = node_val["workflow"]
                        cur_design_html, cur_test_cases_md, cur_test_report_md = get_workflow_panel_data(wf)
                        live_block = render_live_progress(active_stage, subagent_actions, pending_tools)
                        yield _pack(response_text + live_block)

            # 3. 处理 messages 流（主 Agent 的文字输出与工具调用）
            elif mode == "messages":
                chunk, metadata = payload
                node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None

                # 捕获工具调用发起
                if (
                    node in ("model", "agent")
                    and hasattr(chunk, "tool_call_chunks")
                    and chunk.tool_call_chunks
                ):
                    for tc in chunk.tool_call_chunks:
                        if isinstance(tc, dict):
                            name = tc.get("name")
                            args = tc.get("args")
                            if name:
                                current_tool_name = name
                                current_tool_args = ""
                                if name not in pending_tools:
                                    pending_tools.append(name)
                            if args:
                                current_tool_args += args

                # 捕获工具执行完毕 (node == "tools" 或 ToolMessage)
                elif node == "tools" or isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "") or current_tool_name or "tool"
                    tool_args = current_tool_args.strip() or "{}"
                    tool_output = str(chunk.content)

                    if tool_name in pending_tools:
                        pending_tools.remove(tool_name)

                    # 渲染成卡片
                    card_html = render_tool_card(tool_name, tool_args, tool_output)
                    response_text += card_html

                    current_tool_name = ""
                    current_tool_args = ""
                    live_block = render_live_progress(active_stage, subagent_actions, pending_tools)
                    yield _pack(response_text + live_block)

                # 捕获模型的正常文本输出 (主 Agent 对用户的直接回复)
                elif (
                    node in ("model", "agent")
                    and isinstance(chunk, BaseMessage)
                    and chunk.content
                ):
                    if not isinstance(chunk, ToolMessage):
                        token = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                        if token:
                            response_text += token
                            live_block = render_live_progress(active_stage, subagent_actions, pending_tools)
                            yield _pack(response_text + live_block)

    except Exception as e:
        import traceback
        print(f"[ERROR] predict 执行异常: {e}")
        traceback.print_exc()
        response_text += f"\n\n⚠️ **执行出错**: {e}"
        yield _pack(response_text)

    finally:
        # 对话结束后，在后台触发增量索引同步
        async def _safe_bg_sync(target_ws: str):
            try:
                await auto_sync_codebase(target_ws)
            except Exception as sync_err:
                print(f"[Workspace] 后台代码库增量同步异常: {sync_err}")

        asyncio.create_task(_safe_bg_sync(ws))

    # 最终输出（清除临时 live 进度）
    if not response_text:
        response_text = "（智能体已处理完毕）"
    yield _pack(response_text)
