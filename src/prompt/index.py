"""提示词统一出口（各 Agent 的提示词按文件拆分维护）"""

from prompt.common import ENGINEERING_CONSTRAINTS, TOOL_USAGE_RULES, build_workspace_context
from prompt.main_prompt import MAIN_AGENT_SYSTEM_PROMPT
from prompt.design_prompt import DESIGN_AGENT_SYSTEM_PROMPT
from prompt.coding_prompt import CODING_AGENT_SYSTEM_PROMPT
from prompt.testing_prompt import TEST_CASE_AGENT_SYSTEM_PROMPT, TEST_EXEC_AGENT_SYSTEM_PROMPT

# 兼容旧代码：原单 Agent 提示词现由编码 Agent 承担
CODE_AGENT_SYSTEM_PROMPT = CODING_AGENT_SYSTEM_PROMPT

__all__ = [
    "ENGINEERING_CONSTRAINTS",
    "TOOL_USAGE_RULES",
    "build_workspace_context",
    "MAIN_AGENT_SYSTEM_PROMPT",
    "DESIGN_AGENT_SYSTEM_PROMPT",
    "CODING_AGENT_SYSTEM_PROMPT",
    "TEST_CASE_AGENT_SYSTEM_PROMPT",
    "TEST_EXEC_AGENT_SYSTEM_PROMPT",
    "CODE_AGENT_SYSTEM_PROMPT",
]
