"""
主 Agent 的委派工具：把复杂任务的各阶段分派给对应子 Agent

- 子 Agent 的执行细节不暴露给用户：只通过 stream_writer 推送精简进度（custom 流），
  并以 ToolMessage 把阶段结果返回给主 Agent，由主 Agent 统一与用户沟通；
- 每个阶段完成后通过 Command 更新图状态中的 workflow，使进度随会话持久化。
"""

import copy
import os
from typing import Sequence

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from agent.subagents.coding import run_coding_agent
from agent.subagents.design import run_design_agent
from agent.subagents.testing import (
    parse_test_result,
    run_test_case_agent,
    run_test_exec_agent,
    strip_machine_lines,
)
from agent.workflow import (
    design_dir,
    empty_workflow,
    find_design_artifacts,
    tests_dir,
    touch,
)

DELEGATE_DESIGN = "delegate_design"
DELEGATE_CODING = "delegate_coding"
DELEGATE_TESTING = "delegate_testing"
DELEGATION_TOOL_NAMES = (DELEGATE_DESIGN, DELEGATE_CODING, DELEGATE_TESTING)


def _emit(runtime: ToolRuntime, **payload) -> None:
    """向 UI 推送阶段/进度事件（未订阅 custom 流时为空操作）"""
    try:
        if runtime.stream_writer:
            runtime.stream_writer({"source": "workflow", **payload})
    except Exception:
        pass


def _current_workflow(runtime: ToolRuntime) -> dict:
    state = runtime.state or {}
    wf = state.get("workflow") if isinstance(state, dict) else getattr(state, "workflow", None)
    merged = empty_workflow()
    merged.update(copy.deepcopy(wf or {}))
    return merged


def _command(runtime: ToolRuntime, tool_name: str, content: str, wf: dict) -> Command:
    return Command(
        update={
            "workflow": touch(wf),
            "messages": [
                ToolMessage(content=content, tool_call_id=runtime.tool_call_id, name=tool_name)
            ],
        }
    )


def make_delegation_tools(workspace: str, worker_tools: Sequence) -> list:
    """
    :param workspace: 当前项目工作区
    :param worker_tools: 子 Agent 可使用的基础工具（文件系统、命令行、RAG、技能等）
    """
    ws = os.path.abspath(workspace)
    worker_tools = list(worker_tools)

    @tool(DELEGATE_DESIGN)
    async def delegate_design(requirement: str, runtime: ToolRuntime, feedback: str = "") -> Command:
        """【复杂任务 · 阶段 1：方案设计】产出 HTML/图片形式的设计稿，并同步基于设计生成测试用例。
        首次设计只传 requirement；用户对设计提出修改意见时，requirement 传原始需求、feedback 传用户的修改意见。
        返回后必须向用户展示设计要点并等待用户确认，用户确认前不得调用 delegate_coding。

        Args:
            requirement: 完整、具体的需求描述（目标、功能点、约束、技术栈偏好、相关上下文）
            feedback: 用户对上一版设计的修改意见；首次设计时留空
        """
        wf = _current_workflow(runtime)
        is_revision = bool(feedback.strip()) and wf["design_version"] > 0
        previous_artifacts = list(wf.get("design_artifacts") or []) if is_revision else []
        if not is_revision:
            # 新需求：重置工作流，但保留版本号递增，避免覆盖历史设计稿目录
            last_version = wf["design_version"]
            wf = empty_workflow(requirement)
            wf["design_version"] = last_version
        elif requirement.strip():
            wf["requirement"] = requirement.strip()

        version = wf["design_version"] + 1
        out_dir = design_dir(ws, version)

        _emit(runtime, type="stage", stage="design", status="start", text=f"方案设计 v{version}")
        design_res = await run_design_agent(
            workspace=ws,
            tools=worker_tools,
            requirement=wf["requirement"] or requirement,
            version=version,
            output_dir=out_dir,
            feedback=feedback,
            previous_artifacts=previous_artifacts,
            on_progress=lambda t: _emit(runtime, type="progress", stage="design", text=t),
        )
        artifacts = find_design_artifacts(out_dir)
        if not artifacts:
            _emit(runtime, type="stage", stage="design", status="failed")
            content = (
                f"【方案设计失败 · v{version}】未在 `{out_dir}` 中找到设计稿文件。\n"
                f"错误信息：{design_res.error or '无'}\n"
                f"设计阶段输出：\n{design_res.final_text[:2000] or '（无）'}\n\n"
                "【下一步】可调整需求描述后重试，或向用户说明情况。"
            )
            return _command(runtime, DELEGATE_DESIGN, content, wf)

        wf.update(
            stage="awaiting_confirmation",
            design_version=version,
            design_dir=str(out_dir),
            design_artifacts=artifacts,
            design_summary=design_res.final_text,
        )
        _emit(runtime, type="stage", stage="design", status="end", artifacts=artifacts)

        # 设计完成 → 测试 Agent 立即基于设计稿生成测试用例
        cases_path = tests_dir(ws) / f"test_cases_v{version}.md"
        _emit(runtime, type="stage", stage="test_cases", status="start", text="生成测试用例")
        cases_res = await run_test_case_agent(
            workspace=ws,
            tools=worker_tools,
            requirement=wf["requirement"],
            design_artifacts=artifacts,
            design_summary=design_res.final_text,
            output_path=cases_path,
            on_progress=lambda t: _emit(runtime, type="progress", stage="test_cases", text=t),
        )
        if cases_path.exists():
            wf["test_cases_path"] = str(cases_path)
            wf["test_cases_summary"] = cases_res.final_text
            cases_section = f"{cases_res.final_text or '（已生成）'}\n用例文件：`{cases_path}`"
        else:
            cases_section = f"测试用例生成失败：{cases_res.error or '未写入用例文件'}（可在编码后由测试阶段自行梳理用例）"
        _emit(runtime, type="stage", stage="test_cases", status="end")

        artifact_lines = "\n".join(f"- `{p}`" for p in artifacts)
        content = (
            f"【方案设计完成 · v{version}】\n"
            f"设计稿文件：\n{artifact_lines}\n"
            "（设计稿已自动展示在界面右侧「设计稿预览」面板中）\n\n"
            f"## 设计说明\n{design_res.final_text or '（无）'}\n\n"
            f"## 测试用例（已基于设计同步生成）\n{cases_section}\n\n"
            "【下一步】用你自己的话向用户介绍设计要点与测试用例概况，提示用户在右侧面板查看设计稿，"
            "请用户确认设计或提出修改意见；在用户明确确认前不要开始编码。"
        )
        return _command(runtime, DELEGATE_DESIGN, content, wf)

    @tool(DELEGATE_CODING)
    async def delegate_coding(instructions: str, runtime: ToolRuntime, fix_mode: bool = False) -> Command:
        """【复杂任务 · 阶段 2：编码实现】按用户已确认的设计稿完成编码。仅在用户明确确认设计后调用；
        测试未通过需要修复时，设置 fix_mode=true 并在 instructions 中写明失败用例与缺陷。
        编码完成后应立即调用 delegate_testing。

        Args:
            instructions: 实现要求与用户在确认设计时补充的说明；修复模式下为需要修复的缺陷描述
            fix_mode: 是否为根据测试报告修复缺陷的修复模式
        """
        wf = _current_workflow(runtime)
        if not wf.get("design_artifacts"):
            content = "【无法开始编码】当前没有已产出的设计稿。请先调用 delegate_design 完成设计并获得用户确认。"
            return _command(runtime, DELEGATE_CODING, content, wf)

        coding_round = wf["coding_round"] + 1
        label = f"缺陷修复（第 {coding_round} 轮）" if fix_mode else f"编码实现（第 {coding_round} 轮）"
        _emit(runtime, type="stage", stage="coding", status="start", text=label)
        res = await run_coding_agent(
            workspace=ws,
            tools=worker_tools,
            requirement=wf["requirement"],
            instructions=instructions,
            design_artifacts=wf["design_artifacts"],
            design_summary=wf.get("design_summary", ""),
            test_cases_path=wf.get("test_cases_path", ""),
            fix_mode=fix_mode,
            test_report_path=wf.get("test_report_path", "") if fix_mode else "",
            test_summary=wf.get("test_summary", "") if fix_mode else "",
            on_progress=lambda t: _emit(runtime, type="progress", stage="coding", text=t),
        )

        if not res.final_text and res.error:
            _emit(runtime, type="stage", stage="coding", status="failed")
            content = f"【{label}失败】{res.error}\n已执行动作数：{len(res.actions)}。可根据情况重试或向用户说明。"
            return _command(runtime, DELEGATE_CODING, content, wf)

        wf.update(stage="coded", coding_round=coding_round, code_summary=res.final_text, test_passed=None)
        _emit(runtime, type="stage", stage="coding", status="end")
        warn = f"\n\n⚠️ 执行提示：{res.error}" if res.error else ""
        content = (
            f"【{label}完成】\n{res.final_text}{warn}\n\n"
            "【下一步】立即调用 delegate_testing 进行测试验证（无需询问用户）。"
        )
        return _command(runtime, DELEGATE_CODING, content, wf)

    @tool(DELEGATE_TESTING)
    async def delegate_testing(runtime: ToolRuntime, focus: str = "") -> Command:
        """【复杂任务 · 阶段 3：测试验证】依据测试用例对编码结果执行测试并输出测试报告。编码完成后立即调用。

        Args:
            focus: 本轮测试需重点关注的内容（如修复后的回归验证点），可留空
        """
        wf = _current_workflow(runtime)
        test_round = wf["test_round"] + 1
        report_path = tests_dir(ws) / f"test_report_r{test_round}.md"

        _emit(runtime, type="stage", stage="testing", status="start", text=f"测试验证（第 {test_round} 轮）")
        res = await run_test_exec_agent(
            workspace=ws,
            tools=worker_tools,
            requirement=wf["requirement"],
            design_artifacts=wf.get("design_artifacts") or [],
            test_cases_path=wf.get("test_cases_path", ""),
            code_summary=wf.get("code_summary", ""),
            report_path=report_path,
            focus=focus,
            on_progress=lambda t: _emit(runtime, type="progress", stage="testing", text=t),
        )
        passed, stats = parse_test_result(res.final_text)
        summary = strip_machine_lines(res.final_text)

        wf.update(
            stage="completed" if passed else "test_failed",
            test_round=test_round,
            test_report_path=str(report_path) if report_path.exists() else "",
            test_summary=summary,
            test_passed=passed,
            test_stats=stats,
        )
        _emit(runtime, type="stage", stage="testing", status="end" if passed else "failed")

        verdict = "✅ 通过" if passed else ("❌ 未通过" if passed is False else "⚠️ 结果未能解析")
        stats_text = (
            f"（共 {stats['total']} 项，通过 {stats['passed']}，失败 {stats['failed']}）" if stats else ""
        )
        error_text = f"\n执行提示：{res.error}" if res.error else ""
        next_step = (
            "【下一步】向用户做最终交付汇报。"
            if passed
            else "【下一步】若自动修复轮次未超过 2 轮，调用 delegate_coding(fix_mode=true) 修复后重新测试；否则向用户汇报遗留问题。"
        )
        content = (
            f"【测试验证完成 · 第 {test_round} 轮】结果：{verdict}{stats_text}{error_text}\n"
            f"测试报告：`{wf['test_report_path'] or '未生成'}`\n\n{summary or '（无输出）'}\n\n{next_step}"
        )
        return _command(runtime, DELEGATE_TESTING, content, wf)

    return [delegate_design, delegate_coding, delegate_testing]
