import sys
import subprocess
import importlib
from typing import Dict, Set, Optional, Tuple
from tree_sitter import Language


# 1. 语言名称与别名映射到 pip 安装包名与导入模块名
LANGUAGE_PACKAGE_MAP: Dict[str, Tuple[str, str]] = {
    "python": ("tree-sitter-python", "tree_sitter_python"),
    "py": ("tree-sitter-python", "tree_sitter_python"),
    "javascript": ("tree-sitter-javascript", "tree_sitter_javascript"),
    "js": ("tree-sitter-javascript", "tree_sitter_javascript"),
    "jsx": ("tree-sitter-javascript", "tree_sitter_javascript"),
    "typescript": ("tree-sitter-typescript", "tree_sitter_typescript"),
    "ts": ("tree-sitter-typescript", "tree_sitter_typescript"),
    "tsx": ("tree-sitter-typescript", "tree_sitter_typescript"),
    "java": ("tree-sitter-java", "tree_sitter_java"),
    "html": ("tree-sitter-html", "tree_sitter_html"),
    "htm": ("tree-sitter-html", "tree_sitter_html"),
    "go": ("tree-sitter-go", "tree_sitter_go"),
    "rust": ("tree-sitter-rust", "tree_sitter_rust"),
    "rs": ("tree-sitter-rust", "tree_sitter_rust"),
    "cpp": ("tree-sitter-cpp", "tree_sitter_cpp"),
    "c++": ("tree-sitter-cpp", "tree_sitter_cpp"),
    "c": ("tree-sitter-c", "tree_sitter_c"),
    "c_sharp": ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
    "csharp": ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
    "cs": ("tree-sitter-c-sharp", "tree_sitter_c_sharp"),
    "php": ("tree-sitter-php", "tree_sitter_php"),
    "ruby": ("tree-sitter-ruby", "tree_sitter_ruby"),
    "rb": ("tree-sitter-ruby", "tree_sitter_ruby"),
}

# 2. 常见文件后缀映射到语言标识
DEFAULT_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".html": "html",
    ".htm": "html",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".cs": "c_sharp",
    ".php": "php",
    ".rb": "ruby",
    # 兜底纯文本/脚本文件（将回退到行切分）
    ".md": "text",
    ".json": "text",
    ".yaml": "text",
    ".yml": "text",
    ".sql": "text",
}

# 3. 预定义主流语言的 AST 目标节点和容器节点
LANGUAGE_CONFIGS: Dict[str, Dict[str, Set[str]]] = {
    "python": {
        "targets": {
            "function_definition",
            "class_definition",
            "async_function_definition",
            "assignment",
        },
        "containers": {
            "class_definition",
            "module",
        },
    },
    "javascript": {
        "targets": {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
            "lexical_declaration",
            "variable_declaration",
        },
        "containers": {
            "class_declaration",
            "program",
            "class_body",
        },
    },
    "js": {
        "targets": {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "method_definition",
            "arrow_function",
            "lexical_declaration",
            "variable_declaration",
        },
        "containers": {
            "class_declaration",
            "program",
            "class_body",
        },
    },
    "typescript": {
        "targets": {
            "function_declaration",
            "generator_function_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "type_alias_declaration",
            "method_definition",
            "arrow_function",
            "lexical_declaration",
            "variable_declaration",
        },
        "containers": {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "program",
            "class_body",
        },
    },
    "java": {
        "targets": {
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "field_declaration",
            "constant_declaration",
        },
        "containers": {
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
            "class_body",
            "interface_body",
        },
    },
    "html": {
        "targets": {
            "element",
            "script_element",
            "style_element",
        },
        "containers": {
            "element",
            "document",
        },
    },
    "go": {
        "targets": {
            "function_declaration",
            "method_declaration",
            "type_declaration",
            "const_declaration",
            "var_declaration",
        },
        "containers": {
            "type_declaration",
            "source_file",
        },
    },
    "rust": {
        "targets": {
            "function_item",
            "struct_item",
            "enum_item",
            "trait_item",
            "impl_item",
            "const_item",
            "static_item",
            "type_item",
        },
        "containers": {
            "impl_item",
            "trait_item",
            "source_file",
        },
    },
    "cpp": {
        "targets": {
            "function_definition",
            "class_specifier",
            "struct_specifier",
            "enum_specifier",
            "declaration",
            "template_declaration",
        },
        "containers": {
            "class_specifier",
            "struct_specifier",
            "translation_unit",
        },
    },
    "c": {
        "targets": {
            "function_definition",
            "struct_specifier",
            "enum_specifier",
            "declaration",
        },
        "containers": {
            "struct_specifier",
            "translation_unit",
        },
    },
    "c_sharp": {
        "targets": {
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "enum_declaration",
            "property_declaration",
            "field_declaration",
        },
        "containers": {
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "enum_declaration",
            "namespace_declaration",
        },
    },
}

# 4. 全局缓存已实例化的 Tree-sitter Language 对象
_LOADED_LANGUAGES: Dict[str, Language] = {}


def get_or_install_language(language: str, auto_install: bool = True) -> Optional[Language]:
    """
    动态获取或自动下载安装指定语言的 Tree-sitter Parser
    :param language: 语言标识（如 'python', 'java', 'html', 'go'）
    :param auto_install: 若未安装对应包，是否自动通过 pip 安装
    :return: tree_sitter.Language 实例，若失败则返回 None
    """
    if not language:
        return None

    lang_key = language.lower()

    # 1. 优先从内存缓存中获取
    if lang_key in _LOADED_LANGUAGES:
        return _LOADED_LANGUAGES[lang_key]

    if lang_key not in LANGUAGE_PACKAGE_MAP:
        return None

    pkg_name, mod_name = LANGUAGE_PACKAGE_MAP[lang_key]

    # 2. 尝试直接导入已安装的包
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        if not auto_install:
            print(f"[AST] 警告: 未找到 {pkg_name} 模块，跳过 AST 解析")
            return None

        # 3. 动态调用 pip 安装
        print(f"[AST] 检测到未安装语法包，正在自动安装: {pkg_name} ...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg_name],
                check=True,
                capture_output=True,
                text=True,
            )
            # 安装完成后重新加载模块
            mod = importlib.import_module(mod_name)
            print(f"[AST] 成功安装并加载: {pkg_name}")
        except Exception as e:
            print(f"[AST] 自动安装 {pkg_name} 失败: {e}，将回退到行切分模式")
            return None

    # 4. 构建并缓存 Language 对象
    try:
        if lang_key in {"typescript", "ts"} and hasattr(mod, "language_typescript"):
            lang_obj = Language(mod.language_typescript())
        elif lang_key == "tsx" and hasattr(mod, "language_tsx"):
            lang_obj = Language(mod.language_tsx())
        elif hasattr(mod, "language"):
            lang_obj = Language(mod.language())
        else:
            print(f"[AST] 模块 {mod_name} 中未找到 language() 构造方法")
            return None

        _LOADED_LANGUAGES[lang_key] = lang_obj
        return lang_obj
    except Exception as e:
        print(f"[AST] 初始化 {language} Language 对象失败: {e}")
        return None


# 5. 未知/未预设语言的通用启发式规则识别
def is_heuristic_target(node_type: str) -> bool:
    """智能启发式：根据名称模式识别函数、类、结构体等关键业务节点"""
    keywords = ("function", "method", "class", "struct", "interface", "trait", "impl", "enum", "record")
    return any(kw in node_type for kw in keywords) and not node_type.endswith("_body")


def is_heuristic_container(node_type: str) -> bool:
    """智能启发式：识别类、接口、模块、文档等可递归下钻的容器节点"""
    keywords = ("class", "struct", "interface", "impl", "trait", "program", "module", "file", "document", "element")
    return any(kw in node_type for kw in keywords)
