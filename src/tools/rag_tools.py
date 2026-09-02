import os
import re
import sys
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from rag.qdrantCodeHybridRAG import QdrantCodeHybridRAG
from rag.codeContextRetriever import CodeContextRetriever

# 全局单例
_rag_instance: Optional[QdrantCodeHybridRAG] = None
_retriever_instance: Optional[CodeContextRetriever] = None
_current_workspace: Optional[str] = None


def get_collection_name_for_workspace(workspace_path: str) -> str:
    """根据项目目录名称动态生成合法的 Qdrant Collection 名称，例如 'Snake' -> 'snake_kb'"""
    folder_name = Path(workspace_path).resolve().name
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", folder_name).lower().strip("_")
    return f"{safe_name or 'project'}_kb"


def get_rag_service(
    workspace_path: str,
) -> tuple[QdrantCodeHybridRAG, CodeContextRetriever]:
    """根据明确的项目路径获取或初始化该项目的 RAG 服务与检索器单例"""
    global _rag_instance, _retriever_instance, _current_workspace

    if not workspace_path or not os.path.exists(workspace_path):
        raise ValueError("必须指定有效的项目工作区路径！")

    ws = str(Path(workspace_path).resolve())
    qdrant_storage = os.environ.get("QDRANT_PATH") or "./qdrant_data"
    collection_name = get_collection_name_for_workspace(ws)

    if _rag_instance is None or _current_workspace != ws:
        _current_workspace = ws
        _rag_instance = QdrantCodeHybridRAG(
            collection_name=collection_name,
            qdrant_path=qdrant_storage,
        )
        _retriever_instance = CodeContextRetriever(rag=_rag_instance)

    return _rag_instance, _retriever_instance


async def auto_sync_codebase(workspace_path: str, force_full: bool = False) -> dict:
    """
    系统级自动同步：
    - 首次启动项目：全量扫描并构建索引
    - 再次启动/运行时：基于文件 mtime+size 指纹进行毫秒级差异比对，仅对变动文件增量热更新
    """
    if not workspace_path or not os.path.exists(workspace_path):
        return {"status": "invalid_workspace", "path": workspace_path}

    ws = str(Path(workspace_path).resolve())
    rag, _ = get_rag_service(ws)
    sync_result = await rag.sync_directory(repo_dir=ws, force_full=force_full)
    return sync_result


@tool
async def retrieve_code_context(
    query: str,
    target_file: Optional[str] = None,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """
    从当前项目代码知识库中检索相关的业务代码、函数定义以及跨文件引用符号。

    当你需要了解项目架构、查看某个类/函数的实现逻辑、排查 Bug 或分析依赖关系时，请调用此工具。

    Args:
        query: 检索问题或代码功能描述（如 '用户的登录验证逻辑' 或 'Snake 移动与碰撞检测'）
        target_file: 可选，当前关注的特定文件路径（若指定，将优先提取该文件代码并自动分析外部依赖符号）
        start_line: 可选，目标代码起始行号
        end_line: 可选，目标代码结束行号
    """
    global _retriever_instance, _current_workspace
    if not _retriever_instance or not _current_workspace:
        return "当前尚未选定任何项目工作区，无法检索代码知识库。"

    try:
        context_prompt = await _retriever_instance.retrieve_context_for_query(
            query=query,
            target_file=target_file,
            start_line=start_line,
            end_line=end_line,
            top_k=3,
        )
        return context_prompt
    except Exception as e:
        return f"检索代码知识库失败: {str(e)}"
