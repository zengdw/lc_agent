"""
多 Agent 模型配置中心

每个 Agent 角色（主 / 设计 / 编码 / 测试）都可以独立配置模型，优先级：
    1. 界面中保存的覆盖配置（持久化到 data/agent_models.json）
    2. 环境变量  {PREFIX}_PROVIDER / _MODEL / _BASE_URL / _API_KEY / _TEMPERATURE
    3. 默认配置  OPENAI_MODEL / OPENAI_BASE_URL / OPENAI_API_KEY（provider 默认为 openai）
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.chat_models import init_chat_model

AGENT_ROLES: Dict[str, Dict[str, str]] = {
    "main": {"label": "主 Agent", "desc": "负责与用户对话、简单任务直办与复杂任务调度", "env_prefix": "MAIN_AGENT"},
    "design": {"label": "设计 Agent", "desc": "分析需求，产出 HTML / 图片形式的设计稿", "env_prefix": "DESIGN_AGENT"},
    "coding": {"label": "编码 Agent", "desc": "按用户确认的设计稿完成编码实现", "env_prefix": "CODING_AGENT"},
    "testing": {"label": "测试 Agent", "desc": "设计完成后生成测试用例，编码完成后执行测试", "env_prefix": "TEST_AGENT"},
}

CONFIG_FIELDS = ("provider", "model", "base_url", "api_key", "temperature")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(
    os.environ.get("AGENT_MODEL_CONFIG_PATH") or (_PROJECT_ROOT / "data" / "agent_models.json")
)

_lock = threading.Lock()
_model_cache: Dict[tuple, Any] = {}


# ===================== 覆盖配置的读写 =====================
def _load_overrides() -> Dict[str, Dict[str, str]]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[ModelConfig] 读取 {CONFIG_PATH} 失败，忽略覆盖配置: {e}")
        return {}


def get_override(role: str) -> Dict[str, str]:
    """返回界面保存的原始覆盖配置（未填写的字段为空字符串）"""
    raw = _load_overrides().get(role, {}) or {}
    return {f: str(raw.get(f, "") or "") for f in CONFIG_FIELDS}


def save_override(role: str, values: Dict[str, Any]) -> None:
    """保存某个角色的覆盖配置；所有字段为空时等价于清除覆盖"""
    if role not in AGENT_ROLES:
        raise ValueError(f"未知的 Agent 角色: {role}")
    cleaned = {f: str(values.get(f, "") or "").strip() for f in CONFIG_FIELDS}
    if cleaned["temperature"]:
        float(cleaned["temperature"])  # 非法数值直接抛出，由调用方提示
    with _lock:
        data = _load_overrides()
        if any(cleaned.values()):
            data[role] = {k: v for k, v in cleaned.items() if v}
        else:
            data.pop(role, None)
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_override(role: str) -> None:
    save_override(role, {})


# ===================== 生效配置解析 =====================
def _env_default(field: str) -> str:
    return {
        "provider": "openai",
        "model": os.environ.get("OPENAI_MODEL", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "temperature": "",
    }[field]


def get_model_config(role: str) -> Dict[str, str]:
    """按 覆盖配置 > 角色环境变量 > 默认 OPENAI_* 的优先级返回生效配置"""
    if role not in AGENT_ROLES:
        raise ValueError(f"未知的 Agent 角色: {role}")
    prefix = AGENT_ROLES[role]["env_prefix"]
    override = get_override(role)
    resolved = {}
    for field in CONFIG_FIELDS:
        resolved[field] = (
            override.get(field)
            or os.environ.get(f"{prefix}_{field.upper()}", "")
            or _env_default(field)
        ).strip()
    return resolved


def get_inherited_config(role: str) -> Dict[str, str]:
    """忽略界面覆盖时的配置（用于在界面上展示“留空时继承的值”）"""
    prefix = AGENT_ROLES[role]["env_prefix"]
    return {
        f: (os.environ.get(f"{prefix}_{f.upper()}", "") or _env_default(f)).strip()
        for f in CONFIG_FIELDS
    }


def config_signature(role: str) -> tuple:
    cfg = get_model_config(role)
    return tuple(cfg[f] for f in CONFIG_FIELDS)


def describe_model(role: str) -> str:
    cfg = get_model_config(role)
    return f"{cfg['provider']}:{cfg['model'] or '(未配置)'}"


# ===================== 模型实例化 =====================
def create_chat_model(role: str, temperature: Optional[float] = None, proxy: Optional[str] = None):
    """按角色创建（并缓存）聊天模型实例，支持 HTTP/HTTPS 代理配置"""
    cfg = get_model_config(role)
    if not cfg["model"]:
        raise ValueError(
            f"{AGENT_ROLES[role]['label']} 未配置模型名称，请在界面「模型配置」或 .env 中设置 "
            f"{AGENT_ROLES[role]['env_prefix']}_MODEL / OPENAI_MODEL"
        )
    temp = temperature if temperature is not None else (
        float(cfg["temperature"]) if cfg["temperature"] else None
    )

    # 解析有效代理地址：显式传入 > 环境变量 > 未配置时为 None（直连不使用代理）
    raw_proxy = (
        proxy
        or os.getenv("OPENAI_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("http_proxy")
    )
    effective_proxy = raw_proxy.strip() if raw_proxy and raw_proxy.strip() else None

    key = (cfg["provider"], cfg["model"], cfg["base_url"], cfg["api_key"], temp, effective_proxy)
    with _lock:
        if key in _model_cache:
            return _model_cache[key]

    provider = cfg["provider"] or "openai"
    kwargs: Dict[str, Any] = {"model_provider": provider}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    if cfg["api_key"]:
        kwargs["api_key"] = cfg["api_key"]
    if temp is not None:
        kwargs["temperature"] = temp
    if effective_proxy:
        if provider == "openai":
            kwargs["openai_proxy"] = effective_proxy
        else:
            kwargs["proxy"] = effective_proxy

    model = init_chat_model(cfg["model"], **kwargs)
    with _lock:
        _model_cache[key] = model
    return model
