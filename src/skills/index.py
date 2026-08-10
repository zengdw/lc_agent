import os, re, yaml
from typing import Dict
from langchain_core.tools import tool
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"


def parse_skill_meta(filepath: str) -> Dict[str, str]:
    """解析 SKILL.md 中的 YAML Frontmatter 元数据"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if match:
        yaml_text, body = match.groups()
        meta = yaml.safe_load(yaml_text)
        meta["body"] = body
        return meta
    return {
        "name": os.path.basename(os.path.dirname(filepath)),
        "description": "无描述",
        "body": content,
    }


# ================= 定义渐进式披露的核心工具 =================


@tool
def list_available_skills() -> str:
    """获取系统中所有已安装技能的索引目录（仅包含技能名称和简短描述）。
    当你感觉需要特定领域的复杂操作 SOP 或专家指南时，请先调用此工具查看可用技能。
    """
    if not os.path.exists(SKILLS_DIR):
        return "当前未安装任何技能。"

    skills_index = []
    for root, _, files in os.walk(SKILLS_DIR):
        for file in files:
            if file == "SKILL.md":
                meta = parse_skill_meta(os.path.join(root, file))
                skills_index.append(
                    f"- **{meta.get('name')}**: {meta.get('description')}"
                )

    return "系统当前可用的技能目录如下：\n" + "\n".join(skills_index)


@tool
def load_skill(skill_name: str) -> str:
    """根据技能名称从磁盘读取并加载该技能的完整 SOP/执行指南。
    参数:
        skill_name: 技能的唯一标识名称（例如 'github-pr')
    """
    target_path = os.path.join(SKILLS_DIR, skill_name, "SKILL.md")
    if not os.path.exists(target_path):
        return f"错误：未找到名称为 '{skill_name}' 的技能。"

    meta = parse_skill_meta(target_path)
    return f"--- 已成功加载技能 [{skill_name}] 的执行指南 ---\n\n{meta['body']}"


@tool
def get_skill_file_path(skill_name: str, file_path: str) -> str:
    """获取skill相关文件的全路径
    参数:
        skill_name: 技能的唯一标识名称（例如 'github-pr')
        file_path: 文件的相对路径
    """
    target_path = os.path.join(SKILLS_DIR, skill_name, file_path)
    return target_path
