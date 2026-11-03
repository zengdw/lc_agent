"""
多 Agent 协作工作流：状态定义、产物目录约定与状态描述

工作流阶段（workflow["stage"]）：
    idle                  - 无进行中的复杂任务
    awaiting_confirmation - 设计稿与测试用例已产出，等待用户确认
    coded                 - 编码完成，等待/正在测试
    test_failed           - 测试未通过
    completed             - 测试通过，任务交付
"""

import time
from pathlib import Path
from typing import List, Optional

from agent.memory import AgentState

WORKFLOW_DIRNAME = ".agent_workflow"
DESIGN_EXTS = (".html", ".htm", ".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif")
IMAGE_EXTS = (".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif")

STAGE_LABELS = {
    "idle": "空闲",
    "awaiting_confirmation": "等待用户确认设计",
    "coded": "编码完成",
    "test_failed": "测试未通过",
    "completed": "已交付",
}


class OrchestratorState(AgentState):
    """主 Agent 状态：在消息与摘要基础上增加工作流进度"""

    workflow: dict


def empty_workflow(requirement: str = "") -> dict:
    return {
        "stage": "idle",
        "requirement": requirement,
        "design_version": 0,
        "design_dir": "",
        "design_artifacts": [],
        "design_summary": "",
        "test_cases_path": "",
        "test_cases_summary": "",
        "coding_round": 0,
        "code_summary": "",
        "test_round": 0,
        "test_report_path": "",
        "test_summary": "",
        "test_passed": None,
        "test_stats": {},
        "updated_at": "",
    }


def touch(wf: dict) -> dict:
    wf["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return wf


# ===================== 产物目录约定 =====================
def workflow_root(ws: str) -> Path:
    return Path(ws) / WORKFLOW_DIRNAME


def design_dir(ws: str, version: int) -> Path:
    return workflow_root(ws) / "design" / f"v{version}"


def tests_dir(ws: str) -> Path:
    return workflow_root(ws) / "tests"


def find_design_artifacts(directory: Path) -> List[str]:
    """收集设计目录下的设计稿文件（index.html 优先，其余 HTML 次之，图片最后）"""
    if not directory.exists():
        return []
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in DESIGN_EXTS]

    def _rank(p: Path):
        if p.name.lower() == "index.html":
            return (0, str(p))
        if p.suffix.lower() in (".html", ".htm"):
            return (1, str(p))
        return (2, str(p))

    return [str(p) for p in sorted(files, key=_rank)]


def read_text_file(path: str, limit: Optional[int] = None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    if limit and len(text) > limit:
        return text[:limit] + f"\n\n...[已截断，完整内容见 {path}]"
    return text


# ===================== 注入主 Agent 系统提示词的进度描述 =====================
def describe_workflow_for_prompt(wf: Optional[dict]) -> str:
    if not wf or wf.get("stage", "idle") == "idle":
        return (
            "\n\n<workflow_status>\n当前没有进行中的复杂任务工作流。\n</workflow_status>"
        )
    lines = [
        f"- 当前阶段：{wf.get('stage')}（{STAGE_LABELS.get(wf.get('stage'), '')}）",
        f"- 需求：{(wf.get('requirement') or '')[:500]}",
        f"- 设计稿版本：v{wf.get('design_version', 0)}",
    ]
    if wf.get("design_artifacts"):
        lines.append(f"- 设计稿文件：{', '.join(wf['design_artifacts'][:5])}")
    if wf.get("test_cases_path"):
        lines.append(f"- 测试用例：{wf['test_cases_path']}")
    if wf.get("coding_round"):
        lines.append(f"- 已完成编码轮次：{wf['coding_round']}")
    if wf.get("test_round"):
        result = "通过" if wf.get("test_passed") else "未通过"
        lines.append(f"- 已执行测试轮次：{wf['test_round']}，最近一次结果：{result}")
        if wf.get("test_report_path"):
            lines.append(f"- 最新测试报告：{wf['test_report_path']}")
    return "\n\n<workflow_status>\n" + "\n".join(lines) + "\n</workflow_status>"
