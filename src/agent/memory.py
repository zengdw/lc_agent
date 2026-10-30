"""
对话记忆管理：
1. 估算历史消息 token 长度
2. 超过阈值时，将滑动窗口之外的旧消息压缩为摘要（持久化在 state["summary"]），并从 state 中移除
3. 构造模型输入时规范化消息顺序、折叠历史工具输出、截断超长工具输出
"""

import os
import json
from functools import lru_cache
from typing import List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.constants import TAG_NOSTREAM
from langgraph.graph import MessagesState

# ===================== 配置项（可通过环境变量覆盖） =====================
# 历史消息估算 token 超过该值时触发摘要压缩
MAX_HISTORY_TOKENS = int(os.environ.get("MAX_HISTORY_TOKENS", "24000"))
# 压缩后滑动窗口内保留的最近消息 token 预算
KEEP_RECENT_TOKENS = int(os.environ.get("KEEP_RECENT_TOKENS", "8000"))
# 当前轮次中单条工具输出允许的最大 token，超出则在模型输入中截断（不修改 state）
MAX_TOOL_OUTPUT_TOKENS = int(os.environ.get("MAX_TOOL_OUTPUT_TOKENS", "6000"))
# 生成摘要时，单条旧消息送入摘要模型的最大 token
SUMMARY_INPUT_MSG_TOKENS = int(os.environ.get("SUMMARY_INPUT_MSG_TOKENS", "800"))

INTERRUPTED_TOOL_CONTENT = "[执行被用户中断或进程超时，未产生有效输出]"


class AgentState(MessagesState):
    """在 messages 基础上增加滚动摘要字段"""

    summary: str


# ===================== Token 估算 =====================
@lru_cache(maxsize=1)
def _get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    # 启发式回退：CJK 字符约 1 token/字，其他字符约 4 字符/token
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(
                    str(block.get("text", ""))
                    if "text" in block
                    else json.dumps(block, ensure_ascii=False)
                )
        return "\n".join(parts)
    return str(content)


def estimate_message_tokens(msg: BaseMessage) -> int:
    tokens = 4  # 每条消息的角色/格式开销
    tokens += _estimate_text_tokens(_content_to_text(msg.content))
    for tc in getattr(msg, "tool_calls", None) or []:
        tokens += _estimate_text_tokens(tc.get("name", ""))
        tokens += _estimate_text_tokens(
            json.dumps(tc.get("args", {}), ensure_ascii=False)
        )
    return tokens


def estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def _truncate_text(text: str, max_tokens: int) -> str:
    """按 token 预算保留首尾内容，中间省略"""
    if _estimate_text_tokens(text) <= max_tokens:
        return text
    # 粗略按字符比例截取首尾
    ratio = max_tokens / max(_estimate_text_tokens(text), 1)
    keep_chars = max(int(len(text) * ratio), 200)
    head, tail = text[: keep_chars * 2 // 3], text[-keep_chars // 3 :]
    return f"{head}\n\n...[内容过长，已省略约 {len(text) - len(head) - len(tail)} 字符]...\n\n{tail}"


# ===================== 消息规范化 =====================
def _is_tool(m: BaseMessage) -> bool:
    return isinstance(m, ToolMessage) or getattr(m, "type", "") == "tool"


def _is_human(m: BaseMessage) -> bool:
    return getattr(m, "type", "") in ("human", "user")


def normalize_messages(raw_messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """
    严格按 OpenAI 协议规范：任何带 tool_calls 的 AIMessage 紧随其后必须是对应的 ToolMessage。
    修复因中断重试导致的人类消息夹塞 / 缺失的工具结果；同时剔除 state 中的 SystemMessage。
    """
    messages = [m for m in raw_messages if not isinstance(m, SystemMessage)]

    tool_msgs_by_id = {}
    for m in messages:
        if _is_tool(m) and getattr(m, "tool_call_id", None):
            tool_msgs_by_id[m.tool_call_id] = m

    ordered: List[BaseMessage] = []
    used_ids = set()
    for m in messages:
        if _is_tool(m):
            # ToolMessage 统一在其所属 AIMessage 之后排入；
            # 找不到对应 AIMessage 的孤立 ToolMessage 会导致接口报错，直接丢弃
            continue
        ordered.append(m)
        for tc in getattr(m, "tool_calls", None) or []:
            tcid = tc.get("id")
            if not tcid or tcid in used_ids:
                continue
            ordered.append(
                tool_msgs_by_id.get(tcid)
                or ToolMessage(
                    content=INTERRUPTED_TOOL_CONTENT,
                    tool_call_id=tcid,
                    name=tc.get("name", "tool"),
                )
            )
            used_ids.add(tcid)
    return ordered


def _last_human_index(messages: Sequence[BaseMessage]) -> int:
    for idx in range(len(messages) - 1, -1, -1):
        if _is_human(messages[idx]):
            return idx
    return -1


def build_model_input(state: AgentState, system_prompt: str) -> List[BaseMessage]:
    """
    构造送入模型的消息（只影响本次调用，不修改 state）：
    1. SystemMessage = 系统提示词 + 历史摘要
    2. 规范化工具调用顺序
    3. 折叠最新一轮之前的历史 ToolMessage
    4. 截断当前轮次的超长 ToolMessage
    """
    messages = normalize_messages(state.get("messages", []))
    last_user_idx = _last_human_index(messages)

    result: List[BaseMessage] = []
    for idx, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_name = msg.name or "tool"
            text = _content_to_text(msg.content)
            if idx < last_user_idx:
                content = f"[历史工具 `{tool_name}` 输出已折叠，长度: {len(text)} 字符]"
            else:
                content = _truncate_text(text, MAX_TOOL_OUTPUT_TOKENS)
            if content != msg.content:
                msg = ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name,
                    status=getattr(msg, "status", "success"),
                )
        result.append(msg)

    system_content = system_prompt
    summary = state.get("summary") or ""
    if summary:
        system_content += (
            "\n\n<conversation_summary>\n"
            "以下是本会话更早期对话的摘要（原始消息已被压缩移除），请将其作为上下文记忆：\n"
            f"{summary}\n"
            "</conversation_summary>"
        )
    return [SystemMessage(content=system_content)] + result


# ===================== 滑动窗口 + 摘要压缩 =====================
def find_window_start(messages: Sequence[BaseMessage], keep_tokens: int) -> int:
    """
    计算滑动窗口起点（只会切在 HumanMessage 处，保证工具调用对不被拆散）。
    - 从后往前累加，找到最早的、使窗口 token <= keep_tokens 的 HumanMessage
    - 窗口至少包含最后一个 HumanMessage（当前轮次永不被压缩）
    返回 0 表示无需压缩。
    """
    last_user_idx = _last_human_index(messages)
    if last_user_idx <= 0:
        return 0

    start = last_user_idx
    running = estimate_messages_tokens(messages[last_user_idx:])
    for idx in range(last_user_idx - 1, -1, -1):
        running += estimate_message_tokens(messages[idx])
        if running > keep_tokens:
            break
        if _is_human(messages[idx]):
            start = idx
    return start


def _format_for_summary(messages: Sequence[BaseMessage]) -> str:
    lines = []
    for m in messages:
        text = _truncate_text(_content_to_text(m.content), SUMMARY_INPUT_MSG_TOKENS)
        if _is_human(m):
            lines.append(f"[用户]: {text}")
        elif _is_tool(m):
            lines.append(f"[工具 {getattr(m, 'name', '') or 'tool'} 结果]: {text}")
        elif isinstance(m, AIMessage):
            calls = [
                f"{tc.get('name')}({json.dumps(tc.get('args', {}), ensure_ascii=False)[:300]})"
                for tc in (m.tool_calls or [])
            ]
            if text:
                lines.append(f"[助手]: {text}")
            if calls:
                lines.append(f"[助手调用工具]: {'; '.join(calls)}")
    return "\n".join(lines)


SUMMARY_PROMPT = """你是一个对话记忆压缩器。请将【已有摘要】与【新增的历史对话】合并为一份新的精炼摘要，供 AI 编程助手后续继续工作时参考。

要求：
- 保留：用户的核心目标与需求、已做出的关键决策、已修改/创建的文件路径、重要的代码结论、报错及解决方式、尚未完成的待办事项
- 丢弃：寒暄、重复内容、冗长的工具原始输出
- 使用中文，条目化输出，不超过 {max_words} 字
- 只输出摘要正文，不要任何前后缀

【已有摘要】
{summary}

【新增的历史对话】
{conversation}
"""


def get_summary_model():
    """
    创建独立的摘要压缩模型（建议使用更便宜/更快的模型）。
    环境变量（未设置时回退到主模型对应的 OPENAI_* 配置）：
      SUMMARY_MODEL / SUMMARY_BASE_URL / SUMMARY_API_KEY / SUMMARY_TEMPERATURE / SUMMARY_MAX_TOKENS
    """
    from langchain.chat_models import init_chat_model

    model_name = os.environ.get("SUMMARY_MODEL") or os.environ["OPENAI_MODEL"]
    kwargs = {
        "model_provider": "openai",
        "temperature": float(os.environ.get("SUMMARY_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("SUMMARY_MAX_TOKENS", "2048")),
    }
    base_url = os.environ.get("SUMMARY_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("SUMMARY_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return init_chat_model(model_name, **kwargs)


def make_memory_node(model=None):
    """
    创建记忆压缩节点：
    当 state 中消息估算 token 超过 MAX_HISTORY_TOKENS 时，
    把滑动窗口之外的旧消息合并进摘要，并通过 RemoveMessage 从 state（checkpoint）中删除。

    :param model: 摘要模型；为 None 时使用 get_summary_model() 按 SUMMARY_* 环境变量单独创建
    """
    if model is None:
        model = get_summary_model()
    summarizer = model.with_config(tags=[TAG_NOSTREAM], run_name="memory_summarizer")

    async def memory_node(state: AgentState) -> Optional[dict]:
        messages = list(state.get("messages", []))
        total_tokens = estimate_messages_tokens(messages)
        if total_tokens <= MAX_HISTORY_TOKENS:
            return None

        start = find_window_start(messages, KEEP_RECENT_TOKENS)
        if start <= 0:
            return None

        old_messages = messages[:start]
        old_summary = state.get("summary") or "（无）"
        prompt = SUMMARY_PROMPT.format(
            max_words=1500,
            summary=old_summary,
            conversation=_format_for_summary(old_messages),
        )
        try:
            resp = await summarizer.ainvoke([HumanMessage(content=prompt)])
            new_summary = _content_to_text(resp.content).strip()
        except Exception as e:
            print(f"[Memory Warning] 摘要生成失败，本次跳过压缩: {e}")
            return None
        if not new_summary:
            return None

        removable = [RemoveMessage(id=m.id) for m in old_messages if m.id]
        kept_tokens = total_tokens - estimate_messages_tokens(old_messages)
        print(
            f"[Memory] 历史 {total_tokens} tokens 超过阈值 {MAX_HISTORY_TOKENS}，"
            f"压缩 {len(removable)} 条旧消息为摘要，窗口保留约 {kept_tokens} tokens"
        )
        return {"summary": new_summary, "messages": removable}

    return memory_node
