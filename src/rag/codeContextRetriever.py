import os
import re
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient, models


class CodeContextRetriever:
    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedder,  # 前文封装的 Embedding 客户端
        collection_name: str,
    ):
        self.client = qdrant_client
        self.embedder = embedder
        self.collection_name = collection_name

    def retrieve_context_for_query(
        self,
        query: str,
        target_file: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        top_k: int = 3,
    ) -> str:
        """根据提问与可选的代码定位，生成组装好的上下文 Prompt"""

        target_code_block = ""
        referenced_blocks = []

        # 1. 如果指定了具体代码行，优先读取本地文件构建精确上下文
        if target_file and os.path.exists(target_file):
            target_code_block = self._extract_file_scope(
                target_file, start_line, end_line
            )

            # 从目标代码中提取关键调用的符号（简单正则匹配函数调用与类）
            called_symbols = self._extract_identifiers(target_code_block)

            # 精确查出这些符号在其他文件中的定义
            for symbol in called_symbols[:3]:  # 取最核心的前 3 个符号
                symbol_hits = self._search_by_symbol(symbol, exclude_file=target_file)
                referenced_blocks.extend(symbol_hits)

        # 2. 向量检索：召回与问题语义相关的代码
        # 若指定了目标代码，将“问题 + 目标代码摘要”作为查询向量
        search_query = (
            f"{query}\n{target_code_block[:300]}" if target_code_block else query
        )
        dense_vector = self.embedder.embed_query(search_query)

        search_results = self.client.search(
            collection_name=self.collection_name, query_vector=dense_vector, limit=top_k
        )

        # 3. 组装格式化上下文
        formatted_prompt = self._format_prompt(
            user_query=query,
            target_code=target_code_block,
            target_file=target_file,
            start_line=start_line,
            end_line=end_line,
            retrieved_hits=search_results,
            dependency_hits=referenced_blocks,
        )
        return formatted_prompt

    def _extract_file_scope(
        self, file_path: str, start_line: Optional[int], end_line: Optional[int]
    ) -> str:
        """读取指定行及周边上下文"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        total_lines = len(lines)
        s = max(1, start_line if start_line else 1)
        e = min(total_lines, end_line if end_line else total_lines)

        # 向上下各适度扩展 5 行以保留完整逻辑
        context_start = max(1, s - 5)
        context_end = min(total_lines, e + 5)

        selected_lines = lines[context_start - 1 : context_end]
        return f"# File: {file_path} (Lines {context_start}-{context_end})\n" + "".join(
            selected_lines
        )

    def _extract_identifiers(self, code_text: str) -> List[str]:
        """提取代码中的驼峰类名和下划线函数调用"""
        tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{3,})\b", code_text)
        # 过滤常见关键字
        keywords = {
            "def",
            "class",
            "return",
            "import",
            "from",
            "self",
            "async",
            "await",
            "true",
            "false",
            "none",
        }
        return list(set([t for t in tokens if t.lower() not in keywords]))

    def _search_by_symbol(
        self, symbol_name: str, exclude_file: str
    ) -> List[Dict[str, Any]]:
        """按符号精准匹配相关定义"""
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="symbol_name", match=models.MatchValue(value=symbol_name)
                    )
                ],
                must_not=[
                    models.FieldCondition(
                        key="file_path", match=models.MatchValue(value=exclude_file)
                    )
                ],
            ),
            limit=1,
        )[0]
        return [p.payload for p in results]

    def _format_prompt(
        self,
        user_query: str,
        target_code: str,
        target_file: Optional[str],
        start_line: Optional[int],
        end_line: Optional[int],
        retrieved_hits: list,
        dependency_hits: list,
    ) -> str:
        """渲染成结构化清晰的 Prompt 发送给 LLM"""
        prompt = ["你是一个专业的代码助手。请结合提供的代码上下文回答用户的问题。\n"]

        if target_code:
            prompt.append("=== 用户当前关注的目标代码 (Target Code) ===")
            prompt.append(f"```{target_code}```\n")

        if dependency_hits:
            prompt.append(
                "=== 目标代码引用的外部符号与定义 (Resolved Dependencies) ==="
            )
            for dep in dependency_hits:
                prompt.append(
                    f"```\n// File: {dep['file_path']} | Symbol: {dep['symbol_name']}\n{dep['raw_code']}\n```"
                )
            prompt.append("")

        if retrieved_hits:
            prompt.append("=== 向量检索召回的关联业务逻辑 (Retrieved Context) ===")
            seen_files = set()
            for hit in retrieved_hits:
                payload = hit.payload
                # 避免重复展示目标代码
                if (
                    payload.get("file_path") == target_file
                    and payload.get("start_line") == start_line
                ):
                    continue
                file_key = f"{payload.get('file_path')}:{payload.get('symbol_name')}"
                if file_key not in seen_files:
                    seen_files.add(file_key)
                    prompt.append(
                        f"```\n// File: {payload.get('file_path')} | Symbol: {payload.get('symbol_name')}\n{payload.get('raw_code')}\n```"
                    )

        prompt.append(f"\n=== 用户问题 ===\n{user_query}")
        return "\n".join(prompt)
