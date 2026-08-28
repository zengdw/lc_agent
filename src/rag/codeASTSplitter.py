from dataclasses import dataclass
from typing import List, Optional, Set
from tree_sitter import Parser
from astLanguageConfig import (
    get_or_install_language,
    LANGUAGE_CONFIGS,
    is_heuristic_target,
    is_heuristic_container,
)


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str  # 'function', 'class', 'module', 'element', 'variable' 等
    name: str  # 函数名 / 类名 / 标签名
    start_line: int
    end_line: int
    content: str
    context: str  # 文件路径、父类/命名空间、依赖等元数据


class CodeASTSplitter:
    def __init__(
        self,
        language: str,
        max_chunk_size: int = 1500,
        min_variable_chars: int = 100,
        auto_install: bool = True,
        custom_target_nodes: Optional[Set[str]] = None,
    ):
        self.language = (language or "").lower()
        self.max_chunk_size = max_chunk_size
        self.min_variable_chars = min_variable_chars

        # 1. 动态加载或自动安装 Tree-sitter Language 解析器
        lang_obj = get_or_install_language(self.language, auto_install=auto_install)
        if lang_obj:
            self.parser = Parser(lang_obj)
        else:
            self.parser = None

        # 2. 加载 AST 目标节点和容器节点规则
        lang_cfg = LANGUAGE_CONFIGS.get(self.language)
        if custom_target_nodes is not None:
            self.target_node_types = custom_target_nodes
            self.container_node_types = lang_cfg.get("containers", set()) if lang_cfg else set()
        elif lang_cfg:
            self.target_node_types = lang_cfg.get("targets", set())
            self.container_node_types = lang_cfg.get("containers", set())
        else:
            # 未预设的语言将走智能启发式规则判断
            self.target_node_types = None
            self.container_node_types = None

    def _is_target(self, node) -> bool:
        """判断节点是否为核心提取目标"""
        if self.target_node_types is not None:
            return node.type in self.target_node_types
        return is_heuristic_target(node.type)

    def _is_container(self, node) -> bool:
        """判断节点是否为超长时允许递归下钻的容器节点"""
        if self.container_node_types is not None:
            return node.type in self.container_node_types
        return is_heuristic_container(node.type)

    def split_file(self, file_path: str, code_content: str) -> List[CodeChunk]:
        # 1. 如果没有对应的 AST 解析器，直接当作纯文本按行切分
        if self.parser is None:
            return self._fallback_line_split(file_path, code_content, header="")

        # 2. 进行 AST 解析
        code_bytes = code_content.encode("utf-8")
        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node
        chunks: List[CodeChunk] = []

        # 提取文件头部的 import/包声明/head 作为通用上下文
        header_context = self._extract_header_imports(root_node, code_bytes)

        def traverse(node, scope_prefix="", is_top_level=True):
            # 特殊处理 JS/TS 的 export_statement：穿透并保留顶层属性
            if node.type == "export_statement":
                for child in node.children:
                    if child.type != "export":
                        traverse(child, scope_prefix, is_top_level=is_top_level)
                return

            is_target = self._is_target(node)

            # 过滤函数体内部的局部变量 assignment，只保留顶级模块或类属性级别的变量
            if node.type in {"assignment", "lexical_declaration", "variable_declaration"} and not is_top_level:
                is_target = False

            # 命中目标函数、类、标签或顶级变量
            if is_target:
                node_text = code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                name = self._get_node_name(node, code_bytes)
                current_scope = f"{scope_prefix}.{name}" if (scope_prefix and name and name != "document") else (name or scope_prefix)

                # 仅容器节点超长时才递归下钻拆分子节点；函数/方法本身保持完整
                if len(node_text) > self.max_chunk_size and self._is_container(node):
                    for child in node.children:
                        traverse(child, current_scope, is_top_level=False)
                else:
                    context = f"// File: {file_path}\n// Scope: {current_scope}\n{header_context}\n"
                    chunk_type = node.type
                    if node.type in {"function_definition", "function_declaration"} and any(c.type == "async" for c in node.children):
                        chunk_type = "async_function"
                    elif node.type in {
                        "assignment",
                        "lexical_declaration",
                        "variable_declaration",
                        "field_declaration",
                        "constant_declaration",
                        "const_declaration",
                        "var_declaration",
                        "const_item",
                        "static_item",
                    }:
                        chunk_type = "variable"
                        # 过滤无实质语义的短变量
                        if len(node_text.strip()) < self.min_variable_chars:
                            return

                    chunks.append(
                        CodeChunk(
                            file_path=file_path,
                            chunk_type=chunk_type,
                            name=name,
                            start_line=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                            content=node_text,
                            context=context,
                        )
                    )
            else:
                for child in node.children:
                    # 顶级节点包括 root_node 以及包裹 assignment 的 expression_statement / export_statement
                    is_child_top = node == root_node or (
                        node.type in {"module", "program", "source_file", "translation_unit", "expression_statement", "export_statement"}
                        and is_top_level
                    )
                    traverse(child, scope_prefix, is_top_level=is_child_top)

        traverse(root_node, is_top_level=True)

        # 兜底：若文件没有提取出 AST 结构（如纯配置或脚本），回退到滑动窗口分块
        if not chunks:
            return self._fallback_line_split(file_path, code_content, header_context)

        return chunks

    def _get_node_name(self, node, code_bytes: bytes) -> str:
        # 1. HTML Element 提取 tag 和 id/class 标识
        if node.type == "element":
            tag_name = ""
            attr_id = ""
            attr_class = ""
            for child in node.children:
                if child.type in {"start_tag", "self_closing_tag"}:
                    for sub in child.children:
                        if sub.type == "tag_name":
                            tag_name = code_bytes[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace")
                        elif sub.type == "attribute":
                            attr_text = code_bytes[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace")
                            if attr_text.startswith("id="):
                                attr_id = attr_text.split("=", 1)[1].strip("\"'")
                            elif attr_text.startswith("class="):
                                attr_class = attr_text.split("=", 1)[1].strip("\"'")
            if tag_name:
                if attr_id:
                    return f"{tag_name}#{attr_id}"
                elif attr_class:
                    return f"{tag_name}.{attr_class.split()[0]}"
                return tag_name
            return "element"

        if node.type in {"script_element", "style_element"}:
            return node.type.replace("_element", "")

        # 2. Python assignment
        if node.type == "assignment" and node.children:
            return code_bytes[node.children[0].start_byte : node.children[0].end_byte].decode("utf-8", errors="replace").strip()

        # 3. JS / TS / Java / C# 变量与字段声明
        if node.type in {"lexical_declaration", "variable_declaration", "field_declaration"}:
            for child in node.children:
                if child.type == "variable_declarator":
                    for sub in child.children:
                        if sub.type in {"identifier", "name"}:
                            return code_bytes[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace")

        # 4. Go type_declaration / function_declaration / method_declaration
        if node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    for sub in child.children:
                        if sub.type == "type_identifier":
                            return code_bytes[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace")

        # 5. 通用 identifier / name / type_identifier / property_identifier
        for child in node.children:
            if child.type in {"identifier", "name", "type_identifier", "property_identifier", "field_identifier"}:
                return code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        return "anonymous"

    def _extract_header_imports(self, root_node, code_bytes: bytes, max_lines: int = 15) -> str:
        """提取头部 import / package / use / using / include / doctype / head 语句（截取前 N 行）"""
        imports = []
        for child in root_node.children:
            if any(k in child.type for k in ("import", "use", "using", "include", "package", "doctype")):
                imports.append(code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace"))
            elif child.type == "element":
                # HTML <head> 标签抽取
                for sub in child.children:
                    if sub.type == "start_tag" and b"head" in code_bytes[sub.start_byte : sub.end_byte]:
                        head_text = code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                        imports.append("\n".join(head_text.splitlines()[:max_lines]))
                        break
            if len(imports) >= max_lines:
                break
        return "\n".join(imports)

    def _fallback_line_split(
        self, file_path: str, code: str, header: str
    ) -> List[CodeChunk]:
        """对于平铺脚本或配置文件的简单行切分"""
        lines = code.splitlines()
        chunks = []
        step = 50
        for i in range(0, len(lines), step):
            chunk_lines = lines[i : i + step + 10]
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    chunk_type="block",
                    name=f"lines_{i+1}_{i+len(chunk_lines)}",
                    start_line=i + 1,
                    end_line=i + len(chunk_lines),
                    content="\n".join(chunk_lines),
                    context=f"// File: {file_path}\n{header}\n",
                )
            )
        return chunks
