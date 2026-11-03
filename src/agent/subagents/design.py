"""设计 Agent：分析需求，产出 HTML / 图片形式的设计稿"""

from pathlib import Path
from typing import List, Optional, Sequence

from agent.subagents.base import ProgressCallback, SubAgentResult, run_subagent
from prompt.design_prompt import DESIGN_AGENT_SYSTEM_PROMPT


async def run_design_agent(
    *,
    workspace: str,
    tools: Sequence,
    requirement: str,
    version: int,
    output_dir: Path,
    feedback: str = "",
    previous_artifacts: Optional[List[str]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> SubAgentResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    main_file = output_dir / "index.html"

    parts = [
        f"# 设计任务（版本 v{version}）",
        "## 用户需求",
        requirement.strip(),
        "## 输出要求",
        f"- 主设计稿必须写入：`{main_file}`（单文件、自包含 HTML）",
        f"- 补充的 SVG 图片（如有）也保存在目录：`{output_dir}`",
        "- 只允许在上述目录中写文件。",
    ]
    if feedback.strip():
        parts += ["## 用户对上一版设计的修改意见", feedback.strip()]
        if previous_artifacts:
            parts += [
                "## 上一版设计稿（请先读取，在其基础上修改）",
                "\n".join(f"- `{p}`" for p in previous_artifacts),
            ]

    return await run_subagent(
        role="design",
        system_prompt=DESIGN_AGENT_SYSTEM_PROMPT,
        task="\n\n".join(parts),
        tools=tools,
        workspace=workspace,
        on_progress=on_progress,
        recursion_limit=120,
    )
