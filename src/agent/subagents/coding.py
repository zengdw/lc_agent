"""编码 Agent：实现用户确认的设计"""

from typing import List, Optional, Sequence

from agent.subagents.base import ProgressCallback, SubAgentResult, run_subagent
from prompt.coding_prompt import CODING_AGENT_SYSTEM_PROMPT


async def run_coding_agent(
    *,
    workspace: str,
    tools: Sequence,
    requirement: str,
    instructions: str,
    design_artifacts: List[str],
    design_summary: str = "",
    test_cases_path: str = "",
    fix_mode: bool = False,
    test_report_path: str = "",
    test_summary: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> SubAgentResult:
    title = "# 编码任务（修复模式）" if fix_mode else "# 编码任务"
    parts = [
        title,
        "## 用户需求",
        requirement.strip() or "（见设计稿）",
        "## 用户已确认的设计稿（必须先读取并忠实实现）",
        "\n".join(f"- `{p}`" for p in design_artifacts),
    ]
    if design_summary:
        parts += ["## 设计说明", design_summary[:4000]]
    if test_cases_path:
        parts += ["## 验收测试用例（实现需满足这些用例）", f"- `{test_cases_path}`"]
    if instructions.strip():
        parts += ["## 本次实现要求 / 补充说明", instructions.strip()]
    if fix_mode:
        parts += ["## 修复模式说明", "上一轮测试未通过，请根据测试报告定位并修复缺陷。"]
        if test_report_path:
            parts.append(f"- 测试报告：`{test_report_path}`")
        if test_summary:
            parts.append(test_summary[:3000])

    return await run_subagent(
        role="coding",
        system_prompt=CODING_AGENT_SYSTEM_PROMPT,
        task="\n\n".join(parts),
        tools=tools,
        workspace=workspace,
        on_progress=on_progress,
    )
