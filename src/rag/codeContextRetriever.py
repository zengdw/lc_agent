import os
import re
import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import models
from qdrantCodeHybridRAG import QdrantCodeHybridRAG
from astLanguageConfig import DEFAULT_EXT_TO_LANG


class CodeContextRetriever:
    def __init__(self, rag: QdrantCodeHybridRAG):
        self.rag = rag
        self.client = rag.client
        self.collection_name = rag.collection_name

    async def retrieve_context_for_query(
        self,
        query: str,
        target_file: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        top_k: int = 3,
        max_symbols: int = 4,
    ) -> str:
        """
        根据提问与可选的代码定位，结合精准符号依赖查找与混合检索生成高质量上下文 Prompt

        :param query: 用户问题 / 任务描述
        :param target_file: 用户当前聚焦的目标代码文件路径（可选）
        :param start_line: 目标代码起始行号（可选）
        :param end_line: 目标代码结束行号（可选）
        :param top_k: 混合检索召回的代码块数量
        :param max_symbols: 目标代码中最多深入解析的外部依赖符号数量
        :return: 结构化组装好的 Prompt 字符串
        """
        target_code_block = ""
        target_lang = "text"
        referenced_blocks: List[Dict[str, Any]] = []

        # 1. 如果指定了目标文件，提取目标范围代码及周边上下文
        if target_file:
            normalized_target_path = os.path.normpath(target_file)
            if os.path.exists(normalized_target_path):
                target_lang = self._detect_language(normalized_target_path)
                target_code_block = self._extract_file_scope(
                    normalized_target_path, start_line, end_line
                )

                # 提取目标代码中调用的外部符号（函数/类/方法）
                called_symbols = self._extract_identifiers(target_code_block)

                # 并发精准检索被调用符号在其他文件中的定义
                if called_symbols:
                    symbols_to_search = called_symbols[:max_symbols]
                    search_tasks = [
                        self._search_by_symbol(
                            symbol, exclude_file=normalized_target_path
                        )
                        for symbol in symbols_to_search
                    ]
                    symbol_results = await asyncio.gather(
                        *search_tasks, return_exceptions=True
                    )

                    # 收集并去重依赖定义
                    seen_dep_keys = set()
                    for res in symbol_results:
                        if isinstance(res, list):
                            for dep in res:
                                dep_key = (
                                    dep.get("file_path", ""),
                                    dep.get("name", ""),
                                    dep.get("start_line", 0),
                                )
                                if dep_key not in seen_dep_keys:
                                    seen_dep_keys.add(dep_key)
                                    referenced_blocks.append(dep)

        # 2. 混合检索（Dense 语义向量 + Sparse BM25 + RRF 融合重排）
        # 将用户问题与目标代码核心片段组合，强化上下文感知
        if target_code_block:
            summary_code = target_code_block[:400].strip()
            search_query = f"{query}\n[聚焦上下文]:\n{summary_code}"
        else:
            search_query = query

        search_results = await self.rag.hybrid_search(query=search_query, top_k=top_k)

        # 3. 组装格式化上下文 Prompt
        formatted_prompt = self._format_prompt(
            user_query=query,
            target_code=target_code_block,
            target_lang=target_lang,
            target_file=target_file,
            start_line=start_line,
            end_line=end_line,
            retrieved_hits=search_results,
            dependency_hits=referenced_blocks,
        )
        return formatted_prompt

    def _detect_language(self, file_path: str) -> str:
        """根据文件后缀推断代码语言标识（统一使用 DEFAULT_EXT_TO_LANG 配置）"""
        ext = os.path.splitext(file_path)[1].lower()
        return DEFAULT_EXT_TO_LANG.get(ext, "")

    def _extract_file_scope(
        self, file_path: str, start_line: Optional[int], end_line: Optional[int]
    ) -> str:
        """读取指定行及周边上下文，附加行号前缀以增强可读性"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception:
            return ""

        if not lines:
            return ""

        total_lines = len(lines)
        s = max(1, start_line if start_line is not None else 1)
        e = min(total_lines, end_line if end_line is not None else total_lines)
        if s > e:
            s, e = e, s

        # 向上下各适度扩展 3-5 行以保留完整结构
        context_start = max(1, s - 4)
        context_end = min(total_lines, e + 4)

        numbered_lines = []
        for idx in range(context_start, context_end + 1):
            line_content = lines[idx - 1]
            marker = ">> " if (s <= idx <= e) else "   "
            numbered_lines.append(f"{marker}{idx:4d} | {line_content}")

        header = f"# File: {file_path} (Lines {context_start}-{context_end}, Target: {s}-{e})\n"
        return header + "".join(numbered_lines)

    def _extract_identifiers(self, code_text: str) -> List[str]:
        """提取代码中的函数调用、方法调用和类名符号"""
        tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", code_text)
        # 过滤常见编程语言关键字与通用名称
        keywords = {
            "def",
            "class",
            "return",
            "import",
            "from",
            "self",
            "cls",
            "async",
            "await",
            "true",
            "false",
            "none",
            "null",
            "undefined",
            "if",
            "elif",
            "else",
            "for",
            "while",
            "in",
            "is",
            "not",
            "and",
            "or",
            "try",
            "except",
            "finally",
            "with",
            "as",
            "raise",
            "yield",
            "break",
            "continue",
            "pass",
            "lambda",
            "global",
            "nonlocal",
            "assert",
            "var",
            "let",
            "const",
            "function",
            "public",
            "private",
            "protected",
            "static",
            "void",
            "int",
            "str",
            "bool",
            "float",
            "list",
            "dict",
            "set",
        }
        unique_symbols = []
        for t in tokens:
            if t.lower() not in keywords and t not in unique_symbols:
                unique_symbols.append(t)
        return unique_symbols

    async def _search_by_symbol(
        self, symbol_name: str, exclude_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """按符号精准匹配 Qdrant 中的定义节点"""
        must_conditions = [
            models.FieldCondition(
                key="name", match=models.MatchValue(value=symbol_name)
            )
        ]

        scroll_filter = models.Filter(must=must_conditions)

        try:
            scroll_res = await self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=3,
            )
            points = scroll_res[0]
            results = []
            exclude_base = os.path.basename(exclude_file) if exclude_file else None

            for p in points:
                if not p.payload:
                    continue
                payload = p.payload
                file_path = payload.get("file_path", "")

                # 排除同一文件内部的自我引用
                if exclude_file:
                    norm_path = os.path.normpath(file_path)
                    if norm_path == os.path.normpath(exclude_file) or (
                        exclude_base and os.path.basename(norm_path) == exclude_base
                    ):
                        continue

                results.append(payload)
            return results
        except Exception:
            return []

    def _format_prompt(
        self,
        user_query: str,
        target_code: str,
        target_lang: str,
        target_file: Optional[str],
        start_line: Optional[int],
        end_line: Optional[int],
        retrieved_hits: list,
        dependency_hits: list,
    ) -> str:
        """组装结构清晰的高质量 Context Prompt"""
        prompt = [
            "你是一个专业的代码助手。请结合以下提供的代码库上下文，准确、深入地回答用户的问题。\n"
        ]

        # 1. 目标代码区域
        if target_code:
            prompt.append("### 🎯 用户聚焦的目标代码 (Target Code)")
            prompt.append(f"```{target_lang}\n{target_code}\n```\n")

        # 2. 外部依赖符号定义
        if dependency_hits:
            prompt.append("### 🔗 目标代码引用的相关符号与定义 (Resolved Dependencies)")
            for dep in dependency_hits:
                payload = dep.payload if hasattr(dep, "payload") else dep
                file_path = payload.get("file_path", "unknown")
                symbol_name = payload.get("name") or payload.get("symbol_name", "")
                chunk_type = payload.get("chunk_type", "definition")
                code = payload.get("content") or payload.get("raw_code", "")
                lang = self._detect_language(file_path)

                prompt.append(f"**[{chunk_type}] `{symbol_name}`** (`{file_path}`):")
                prompt.append(f"```{lang}\n{code}\n```\n")

        # 3. 混合检索召回的上下文
        if retrieved_hits:
            prompt.append("### 📚 混合检索召回的关联业务逻辑 (Retrieved Context)")
            seen_items = set()
            target_norm = os.path.normpath(target_file) if target_file else None

            for hit in retrieved_hits:
                payload = hit.payload if hasattr(hit, "payload") else hit
                file_path = payload.get("file_path", "")
                norm_hit_file = os.path.normpath(file_path) if file_path else ""

                # 避免重复展示与目标代码完全重叠的分块
                hit_start = payload.get("start_line")
                hit_end = payload.get("end_line")
                if (
                    target_norm
                    and norm_hit_file == target_norm
                    and start_line is not None
                    and end_line is not None
                    and hit_start is not None
                    and hit_end is not None
                ):
                    # 检查是否有重叠
                    if not (hit_end < start_line or hit_start > end_line):
                        continue

                symbol_name = payload.get("name") or payload.get("symbol_name", "")
                code = payload.get("content") or payload.get("raw_code", "")
                score = (
                    hit.get("score")
                    if isinstance(hit, dict)
                    else getattr(hit, "score", None)
                )
                score_str = f" | RRF Score: {score:.4f}" if score is not None else ""
                lang = self._detect_language(file_path)

                item_key = f"{file_path}:{symbol_name}:{hit_start}"
                if item_key not in seen_items:
                    seen_items.add(item_key)
                    line_info = f"Lines {hit_start}-{hit_end}" if hit_start else ""
                    header_info = f"`{file_path}` {line_info} {score_str}".strip()
                    prompt.append(f"**文件/符号**: {header_info}")
                    prompt.append(f"```{lang}\n{code}\n```\n")

        # 4. 用户问题
        prompt.append("---")
        prompt.append(f"### ❓ 用户问题\n{user_query}")
        return "\n".join(prompt)
