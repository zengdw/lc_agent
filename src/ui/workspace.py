"""工作区选择、索引同步与状态横幅"""

import os

from agent.session_manager import update_session_workspace
from tools.rag_tools import auto_sync_codebase
from ui.resources import build_status_html
from ui.state import app_state


def get_workspace_status_for_session(ws_path: str) -> tuple[str, str]:
    """根据会话绑定的工作区路径返回 (workspace_input_val, status_html)"""
    target = (ws_path or "").strip()
    if not target:
        init_html = build_status_html(
            "warning",
            "尚未绑定项目工作区",
            "请点击上方「📂 浏览选择文件夹」绑定本地项目根目录后开始交互；支持在对话框中发送 /new 重置会话。",
        )
        return "", init_html
    if os.path.exists(target):
        detail = f"已恢复当前会话绑定的工作区：`{target}`。智能体已热就绪，可以直接开始对话交互！"
        return target, build_status_html("success", "代码库知识库已就绪", detail)
    else:
        detail = f"该会话原绑定的工作区路径不存在或已被移动：`{target}`，请重新选择本地项目根目录。"
        return target, build_status_html("warning", "工作区路径未找到", detail)


def open_folder_dialog(current_path: str) -> str:
    """唤起系统原生文件夹选择对话框并返回真实绝对路径"""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # 保持弹窗置顶
        initial_dir = (
            current_path
            if (current_path and os.path.exists(current_path))
            else os.path.expanduser("~")
        )
        selected_dir = filedialog.askdirectory(
            initialdir=initial_dir, title="选择项目工作区文件夹"
        )
        root.destroy()
        return selected_dir if selected_dir else current_path
    except Exception as e:
        print(f"[警告] 唤起文件夹选择对话框失败: {e}")
        return current_path


def _describe_sync_result(sync_res: dict, first_time: bool) -> tuple[str, str, str]:
    """返回 (status_type, title, detail)"""
    status = sync_res.get("status")
    total = sync_res.get("total_files", 0)
    if status == "empty_workspace":
        return (
            "info",
            "工作区已就绪（空白项目）",
            "当前工作区为空（未检测到代码文件）。你可以直接向智能体提问，从零开始搭建和编写新项目！",
        )
    if status == "full_indexed":
        detail = f"{'首次' if first_time else ''}全量索引构建完成！共成功索引 {total} 个工程代码文件。"
    elif status == "no_change":
        detail = f"代码索引校验通过：代码库无变更，智能体已热就绪（共 {total} 个代码文件）。"
    elif status == "incrementally_synced":
        detail = (
            f"增量同步完成！新增: {sync_res.get('added', 0)}, "
            f"修改: {sync_res.get('modified', 0)}, 删除: {sync_res.get('deleted', 0)}。"
        )
    else:
        detail = f"索引处理完成：{sync_res}"
    title = "代码库知识库已就绪" if first_time else "知识库重新同步完成"
    return "success", title, detail


async def _sync_and_bind(target: str, first_time: bool) -> str:
    sync_res = await auto_sync_codebase(target)
    # 预热并初始化该工作区 Agent，同时更新当前会话工作区记录
    app_state.current_workspace = target
    update_session_workspace(app_state.current_thread_id, target)
    await app_state.get_cached_agent(target)
    status_type, title, detail = _describe_sync_result(sync_res, first_time)
    print(f"[Workspace] {detail}\n")
    return build_status_html(status_type, title, detail)


async def handle_folder_selection_and_index(current_path: str) -> tuple[str, str]:
    """选择文件夹并在选定后立即触发该项目的索引构建/指纹增量同步"""
    selected_dir = open_folder_dialog(current_path)
    if not selected_dir or not os.path.exists(selected_dir):
        return current_path, build_status_html(
            "warning",
            "未选择有效文件夹",
            "请重新点击按钮并选择本地代码项目所在的根目录文件夹。",
        )

    print(f"\n[Workspace] 用户选定文件夹: {selected_dir}，开始执行索引同步...")
    try:
        return selected_dir, await _sync_and_bind(selected_dir, first_time=True)
    except Exception as e:
        import traceback

        err_msg = f"索引构建失败: {str(e)}"
        print(f"[Workspace] {err_msg}")
        traceback.print_exc()
        return selected_dir, build_status_html("danger", "代码索引异常", err_msg)


async def handle_manual_sync(path: str) -> str:
    """手动点击或输入框修改后触发索引同步"""
    target = (path or "").strip()
    if not target or not os.path.exists(target):
        return build_status_html(
            "warning",
            "指定路径不存在",
            "请先输入或选择有效的本地项目路径后再进行同步。",
        )
    try:
        return await _sync_and_bind(target, first_time=False)
    except Exception as e:
        return build_status_html("danger", "知识库同步失败", str(e))
