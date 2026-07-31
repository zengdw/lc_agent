import subprocess, time
from langchain_core.tools import tool


MAX_OUTPUT_LENGTH = 20000


@tool
def run_shell_command(command: str, cwd: str = None) -> str:
    """在系统的终端 Shell 中执行给定的 Bash/Shell 命令，并返回标准输出或错误输出。

    Args:
        command: 需要执行的命令
        cwd: 命令执行的工作目录，默认为 None
    """
    try:
        # 执行命令并捕获输出
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 设置超时防止无限挂起
        )
        output = result.stdout if result.returncode == 0 else result.stderr
        if not output:
            return "命令执行成功，无输出。"

        output = output.strip()
        if len(output) > MAX_OUTPUT_LENGTH:
            truncated_len = len(output) - MAX_OUTPUT_LENGTH
            output = (
                output[:MAX_OUTPUT_LENGTH]
                + f"\n\n[警告：输出过长 ({len(output)} 字符)，已截断末尾 {truncated_len} 字符]"
            )
        return output
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（超过 30 秒）。"
    except Exception as e:
        return f"执行失败：{str(e)}"


@tool
def get_current_time():
    """获取当前时间"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
