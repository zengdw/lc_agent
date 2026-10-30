import os, signal, subprocess, tempfile, time, locale
from langchain_core.tools import tool


MAX_OUTPUT_LENGTH = 20000
DEFAULT_TIMEOUT = 30
MAX_TIMEOUT = 300
IS_WINDOWS = os.name == "nt"


def _decode(data: bytes) -> str:
    """优先 UTF-8，失败则回退系统编码（Windows 中文环境下 cmd 输出为 GBK）"""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False) or "gbk", errors="replace")


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """杀掉 shell 及其派生的全部子进程（仅 kill shell 会遗留孙进程）"""
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def _read_tmp(f) -> str:
    try:
        f.seek(0)
        return _decode(f.read())
    except Exception:
        return ""


@tool
def run_shell_command(command: str, cwd: str = None, timeout: int = DEFAULT_TIMEOUT) -> str:
    """在系统的终端 Shell 中执行给定的命令，并返回退出码、标准输出与错误输出。
    注意：命令不能等待交互输入（stdin 已关闭）；如需启动长期运行的服务（如 http.server、npm run dev），
    请让其在后台运行（Windows: `start /b ...`，Linux: `nohup ... &`），否则会在超时后被强制终止。
    Windows 下需要延时等待时不要用 `timeout /t N`（stdin 关闭时会立即报错），请用 `ping -n N+1 127.0.0.1 >nul`。

    Args:
        command: 需要执行的命令
        cwd: 命令执行的工作目录，默认为 None
        timeout: 超时时间（秒），默认 30，最大 300
    """
    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))

    # 关键：输出写入临时文件而不是管道。
    # 使用管道时，命令派生的后台进程会继承管道句柄且不关闭，导致读取方永远等不到 EOF 而卡死。
    out_f = tempfile.TemporaryFile()
    err_f = tempfile.TemporaryFile()
    popen_kwargs = dict(
        cwd=cwd,
        shell=True,
        stdin=subprocess.DEVNULL,  # 防止命令等待交互输入而挂起
        stdout=out_f,
        stderr=err_f,
    )
    if not IS_WINDOWS:
        popen_kwargs["start_new_session"] = True  # 便于超时后按进程组整体清理

    try:
        proc = subprocess.Popen(command, **popen_kwargs)
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc)
            try:
                returncode = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                returncode = None

        stdout, stderr = _read_tmp(out_f).strip(), _read_tmp(err_f).strip()
    except Exception as e:
        return f"执行失败：{str(e)}"
    finally:
        for f in (out_f, err_f):
            try:
                f.close()
            except Exception:
                pass

    parts = []
    if timed_out:
        parts.append(f"错误：命令执行超时（超过 {timeout} 秒），已强制终止进程树。")
    else:
        parts.append(f"退出码: {returncode}")
    if stdout:
        parts.append(f"[stdout]\n{stdout}")
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if not stdout and not stderr and not timed_out:
        parts.append("命令执行完成，无输出。")

    output = "\n".join(parts)
    if len(output) > MAX_OUTPUT_LENGTH:
        truncated_len = len(output) - MAX_OUTPUT_LENGTH
        output = (
            output[:MAX_OUTPUT_LENGTH]
            + f"\n\n[警告：输出过长 ({len(output)} 字符)，已截断末尾 {truncated_len} 字符]"
        )
    return output


@tool
def get_current_time():
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
