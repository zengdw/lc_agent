"""测试 Agent：设计完成时生成测试用例；编码完成后执行测试并输出测试报告"""

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agent.subagents.base import ProgressCallback, SubAgentResult, run_subagent
from prompt.testing_prompt import TEST_CASE_AGENT_SYSTEM_PROMPT, TEST_EXEC_AGENT_SYSTEM_PROMPT


async def run_test_case_agent(
    *,
    workspace: str,
    tools: Sequence,
    requirement: str,
    design_artifacts: List[str],
    design_summary: str,
    output_path: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> SubAgentResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task = "\n\n".join(
        [
            "# 测试用例编写任务",
            "## 用户需求",
            requirement.strip(),
            "## 设计稿文件（请先读取）",
            "\n".join(f"- `{p}`" for p in design_artifacts),
            "## 设计说明",
            design_summary[:4000] or "（见设计稿）",
            "## 输出要求",
            f"- 测试用例写入：`{output_path}`",
        ]
    )
    return await run_subagent(
        role="testing",
        system_prompt=TEST_CASE_AGENT_SYSTEM_PROMPT,
        task=task,
        tools=tools,
        workspace=workspace,
        on_progress=on_progress,
        recursion_limit=80,
    )


async def run_test_exec_agent(
    *,
    workspace: str,
    tools: Sequence,
    requirement: str,
    design_artifacts: List[str],
    test_cases_path: str,
    code_summary: str,
    report_path: Path,
    focus: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> SubAgentResult:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# 测试执行任务",
        "## 用户需求",
        requirement.strip() or "（见设计稿）",
        "## 测试用例文件",
        f"- `{test_cases_path}`" if test_cases_path else "（暂无用例文件，请依据设计稿与代码梳理核心用例）",
        "## 设计稿文件",
        "\n".join(f"- `{p}`" for p in design_artifacts) or "（无）",
        "## 编码阶段的实现说明",
        code_summary[:4000] or "（无）",
    ]
    if focus.strip():
        parts += ["## 本轮重点关注", focus.strip()]
    parts += ["## 输出要求", f"- 测试报告写入：`{report_path}`"]

    return await run_subagent(
        role="testing",
        system_prompt=TEST_EXEC_AGENT_SYSTEM_PROMPT,
        task="\n\n".join(parts),
        tools=tools,
        workspace=workspace,
        on_progress=on_progress,
    )


_RESULT_RE = re.compile(r"TEST_RESULT\s*[:：]\s*(PASS|FAIL)", re.IGNORECASE)
_STATS_RE = re.compile(
    r"total\s*=\s*(\d+)\s+passed\s*=\s*(\d+)\s+failed\s*=\s*(\d+)", re.IGNORECASE
)


def parse_test_result(text: str) -> Tuple[Optional[bool], dict]:
    """解析测试 Agent 最终回复中的 TEST_RESULT 与 STATS 行；无法解析时 passed 为 None"""
    passed: Optional[bool] = None
    stats: dict = {}
    matches = _RESULT_RE.findall(text or "")
    if matches:
        passed = matches[-1].upper() == "PASS"
    stats_match = _STATS_RE.findall(text or "")
    if stats_match:
        total, ok, failed = (int(x) for x in stats_match[-1])
        stats = {"total": total, "passed": ok, "failed": failed}
        if passed is None:
            passed = failed == 0 and total > 0
    return passed, stats


def strip_machine_lines(text: str) -> str:
    """去掉仅供系统解析的 STATS / TEST_RESULT 行"""
    lines = [
        line for line in (text or "").splitlines()
        if not _RESULT_RE.search(line) and not line.strip().upper().startswith("STATS:")
    ]
    return "\n".join(lines).strip()
