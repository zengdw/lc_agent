"""前端静态资源加载与通用状态横幅"""

import os

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_resource(*path_segments: str) -> str:
    """按相对 src/ 的路径加载前端资源文件 (CSS / JS / HTML)"""
    full_path = os.path.join(_SRC_DIR, *path_segments)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


STATUS_BANNER_TEMPLATE = load_resource("templates", "status_banner.html")


def build_status_html(status_type: str, title: str, desc: str) -> str:
    """构建具有状态呼吸灯和高颜值现代卡片质感的状态 HTML"""
    icon_map = {"warning": "⚠️", "success": "✅", "danger": "❌", "info": "ℹ️"}
    icon = icon_map.get(status_type, "ℹ️")
    return STATUS_BANNER_TEMPLATE.format(
        status_type=status_type,
        icon=icon,
        title=title,
        desc=desc,
    )
