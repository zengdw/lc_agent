from dataclasses import dataclass
from typing import List, Optional
from tree_sitter import Language, Parser
import tree_sitter_python as tspython


# 构建语言映射表
LANGUAGES = {
    "python": Language(tspython.language()),
}


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str  # 'function', 'class', 'module'
    name: str  # 函数名 / 类名
    start_line: int
    end_line: int
    content: str
    context: str  # 文件路径、父类/命名空间、依赖等元数据


class CodeASTSplitter:
    def __init__(self, language: str, max_chunk_size: int = 1500, min_variable_chars: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_variable_chars = min_variable_chars
        lang_obj = LANGUAGES.get(language)
        if lang_obj:
            self.parser = Parser(lang_obj)
        else:
            self.parser = None

        # 定义需要提取的目标语法节点类型
        self.target_node_types = {
            "python": {
                "function_definition",
                "class_definition",
                "async_function_definition",
                "assignment",
            },
            "typescript": {
                "function_declaration",
                "class_declaration",
                "method_definition",
                "arrow_function",
                "lexical_declaration",
                "variable_declaration",
            },
            "java": {
                "method_declaration",
                "class_declaration",
                "interface_declaration",
                "field_declaration",
            },
        }.get(language, {"function_definition", "class_definition"})

        # 结构性容器节点类型（仅这些节点过长时允许向下拆解内部方法/类）
        self.container_node_types = {
            "class_definition",
            "class_declaration",
            "interface_declaration",
            "module",
            "program",
        }

    def split_file(self, file_path: str, code_content: str) -> List[CodeChunk]:
        # 1. 如果没有对应的 AST 解析器，直接当作纯文本按行切分
        if self.parser is None:
            return self._fallback_line_split(file_path, code_content, header="")

        # 2. 进行 AST 解析
        code_bytes = code_content.encode("utf-8")
        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node
        chunks: List[CodeChunk] = []

        # 提取文件头部的 import/包声明作为通用上下文
        header_context = self._extract_header_imports(root_node, code_bytes)

        def traverse(node, scope_prefix="", is_top_level=True):
            is_target = node.type in self.target_node_types

            # 过滤函数体内部的局部变量 assignment，只保留顶级模块或类属性级别的变量
            if node.type in {"assignment", "lexical_declaration", "variable_declaration"} and not is_top_level:
                is_target = False

            # 命中目标函数、类或顶级变量
            if is_target:
                node_text = code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                name = self._get_node_name(node, code_bytes)
                current_scope = f"{scope_prefix}.{name}" if scope_prefix else name

                # 仅容器节点（如超长大类、接口）超长时才递归下钻拆分为方法；函数/方法本身（包含闭包辅助函数）保持完整
                if len(node_text) > self.max_chunk_size and node.type in self.container_node_types:
                    for child in node.children:
                        traverse(child, current_scope, is_top_level=False)
                else:
                    context = f"// File: {file_path}\n// Scope: {current_scope}\n{header_context}\n"
                    chunk_type = node.type
                    if node.type == "function_definition" and any(c.type == "async" for c in node.children):
                        chunk_type = "async_function_definition"
                    elif node.type in {"assignment", "lexical_declaration", "variable_declaration", "field_declaration"}:
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
                    # 顶级节点包括 root_node 以及包裹 assignment 的 expression_statement
                    is_child_top = node == root_node or (node.type in {"module", "program", "expression_statement"} and is_top_level)
                    traverse(child, scope_prefix, is_top_level=is_child_top)

        traverse(root_node, is_top_level=True)

        # 兜底：若文件没有提取出 AST 结构（如纯配置或脚本），回退到滑动窗口分块
        if not chunks:
            return self._fallback_line_split(file_path, code_content, header_context)

        return chunks

    def _get_node_name(self, node, code_bytes: bytes) -> str:
        if node.type == "assignment" and node.children:
            return code_bytes[node.children[0].start_byte : node.children[0].end_byte].decode("utf-8", errors="replace").strip()
        for child in node.children:
            if child.type in {"identifier", "name"}:
                return code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        return "anonymous"

    def _extract_header_imports(self, root_node, code_bytes: bytes, max_lines: int = 15) -> str:
        """提取头部 import / using 语句（截取前 N 行）"""
        imports = []
        for child in root_node.children:
            if "import" in child.type or "use" in child.type or "package" in child.type:
                imports.append(code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace"))
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
