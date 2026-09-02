import uuid
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient, models
from fastembed import SparseTextEmbedding
from rag.codebaseLoader import CodebaseLoader
from rag.codeASTSplitter import CodeASTSplitter
from rag.customOpenAIEmbedding import CustomOpenAIEmbedding


class QdrantCodeHybridRAG:
    def __init__(
        self,
        collection_name: str,
        qdrant_path: str,  # 本地持久化目录，或传 url="http://localhost:6333"
    ):
        self.collection_name = collection_name
        self.qdrant_path = qdrant_path

        # 1. 初始化异步客户端与向量模型
        self.client = AsyncQdrantClient(path=qdrant_path)
        self.dense_model = CustomOpenAIEmbedding()
        # BM25 稀疏模型 (FastEmbed 本地分词与权重计算)
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        self._collection_initialized = False

    async def setup_collection(self):
        """异步定义包含 Dense 和 Sparse 两种向量命名的 Schema"""
        if not await self.client.collection_exists(self.collection_name):
            dense_dim = await self.dense_model.aget_dimension()
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"已成功创建集合: {self.collection_name}")

            # 仅在连接远程 Qdrant Server 时才需要显式创建 payload 索引（本地模式自带遍历过滤，无需创建）
            if not self.qdrant_path:
                try:
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="file_path",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

        self._collection_initialized = True

    async def delete_by_file_path(self, file_path: str):
        """物理删除指定文件路径下的所有旧 Chunks"""
        if not self._collection_initialized:
            await self.setup_collection()

        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_path", match=models.MatchValue(value=file_path)
                        )
                    ]
                )
            ),
        )
        print(f"[清理完成] 已移除 {file_path} 的所有历史数据")

    async def index_code_chunks(self, code_chunks: List[Dict], batch_size: int = 64):
        """
        异步分批计算 Dense 与 Sparse 向量并写入 Qdrant（高吞吐批量模式）
        """
        if not code_chunks:
            return

        if not self._collection_initialized:
            await self.setup_collection()

        total_chunks = len(code_chunks)
        points_count = 0

        for i in range(0, total_chunks, batch_size):
            batch = code_chunks[i : i + batch_size]
            texts = [
                f"{chunk.get('context', '')}\n{chunk.get('content', '')}"
                for chunk in batch
            ]

            # 1. 批量异步请求 OpenAI 接口计算 Dense 向量
            dense_embeddings = await self.dense_model.aembed_documents(texts)

            # 2. 批量计算 Sparse (BM25) 向量（FastEmbed 为 CPU 密集型，投递到线程池避免阻塞事件循环）
            sparse_embeddings = await asyncio.to_thread(
                lambda: list(self.sparse_model.embed(texts))
            )

            # 3. 构造 PointStruct
            points = []
            for j, chunk in enumerate(batch):
                sparse_val = sparse_embeddings[j]
                sparse_vector = models.SparseVector(
                    indices=sparse_val.indices.tolist(),
                    values=sparse_val.values.tolist(),
                )

                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense_vector": dense_embeddings[j],
                        "sparse_vector": sparse_vector,
                    },
                    payload=chunk,
                )
                points.append(point)

            # 4. 原生异步写入 Qdrant
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            points_count += len(points)
            print(f"-> 已写入 Qdrant 进度: {points_count}/{total_chunks} 个分块...")

        print(f"✅ 成功完成全部 {points_count} 个代码片段的索引构建！")

    # 兼容别名
    async def index_directory(
        self, repo_dir: str, clean_existing: bool = True, batch_size: int = 64
    ):
        """
        一键遍历本地代码库并构建/更新混合索引（全异步流程）
        """
        loader = CodebaseLoader(base_dir=repo_dir)
        print(f"正在扫描并切分目录: {repo_dir} ...")
        all_chunks = []
        for file_path, file_chunks in loader.load_and_split_all():
            if clean_existing:
                await self.delete_by_file_path(file_path)
            all_chunks.extend(file_chunks)

        if not all_chunks:
            print("未找到需要索引的代码文件。")
            return

        # 批量构建索引
        await self.index_code_chunks(all_chunks, batch_size=batch_size)
        print(f"代码库 {repo_dir} 索引构建完成！")

    async def index_file(
        self, file_path: str, base_dir: Optional[str] = None, max_chunk_size: int = 1500
    ):
        """
        单文件增量热更新：物理清理该文件旧向量并重新 AST 切分与写入
        """
        path_obj = Path(file_path).resolve()
        if base_dir:
            base_path = Path(base_dir).resolve()
            rel_path = path_obj.relative_to(base_path).as_posix()
        else:
            rel_path = str(file_path).replace("\\", "/")

        # 1. 物理删除旧数据
        await self.delete_by_file_path(rel_path)

        if not path_obj.exists() or not path_obj.is_file():
            return

        # 2. 读取并切分
        ext = path_obj.suffix.lower()
        from astLanguageConfig import DEFAULT_EXT_TO_LANG

        lang = DEFAULT_EXT_TO_LANG.get(ext)
        content = path_obj.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return

        splitter = CodeASTSplitter(language=lang, max_chunk_size=max_chunk_size)
        chunks = splitter.split_file(file_path=rel_path, code_content=content)
        if chunks:
            chunk_dicts = [c.__dict__ if hasattr(c, "__dict__") else c for c in chunks]
            await self.index_code_chunks(chunk_dicts)

    async def sync_directory(
        self, repo_dir: str, force_full: bool = False, batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        自动比对文件指纹（mtime + size）进行智能同步：
        - 首次启动：全量扫描并切分构建，保存指纹元数据
        - 再次启动：比对差异，无变更则直接跳过（0开销），有变动则仅增量热更新变更文件
        """
        repo_path = Path(repo_dir).resolve()
        if not repo_path.exists():
            return {"status": "dir_not_found", "repo_dir": repo_dir}

        # 确定指纹元数据存储文件（存放在 qdrant_path 内部，避免污染用户项目）
        meta_file = Path(self.qdrant_path) / f"{self.collection_name}_files_meta.json"
        cached_meta: Dict[str, Dict[str, Any]] = {}
        if meta_file.exists() and not force_full:
            try:
                cached_meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:
                cached_meta = {}

        # 扫描工作区有效代码文件
        loader = CodebaseLoader(base_dir=str(repo_path))
        current_files: Dict[str, Dict[str, Any]] = {}
        for file_path in loader.walk_files():
            rel = file_path.relative_to(repo_path).as_posix()
            try:
                stat = file_path.stat()
                current_files[rel] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                    "abs_path": str(file_path),
                }
            except Exception:
                continue

        # 1. 首次启动或强制全量构建
        if not cached_meta or force_full:
            print(f"[RAG 知识库] 首次启动/全量更新：正在为 {repo_dir} 构建索引...")
            await self.index_directory(
                repo_dir=str(repo_path), clean_existing=True, batch_size=batch_size
            )
            # 持久化文件指纹缓存
            try:
                meta_file.parent.mkdir(parents=True, exist_ok=True)
                meta_file.write_text(
                    json.dumps(current_files, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[RAG 知识库] 保存元数据指纹失败: {e}")

            return {
                "status": "full_indexed",
                "total_files": len(current_files),
            }

        # 2. 再次启动：差异比对
        cur_keys = set(current_files.keys())
        old_keys = set(cached_meta.keys())

        added_keys = cur_keys - old_keys
        deleted_keys = old_keys - cur_keys
        modified_keys = {
            k
            for k in (cur_keys & old_keys)
            if current_files[k]["mtime"] != cached_meta[k].get("mtime")
            or current_files[k]["size"] != cached_meta[k].get("size")
        }

        changed_count = len(added_keys) + len(modified_keys) + len(deleted_keys)
        if changed_count == 0:
            print(
                f"[RAG 知识库] 校验完成：代码库无变更，跳过索引（共 {len(current_files)} 个文件）。"
            )
            return {"status": "no_change", "total_files": len(current_files)}

        print(
            f"[RAG 知识库] 检测到代码变动 (新增: {len(added_keys)}, 修改: {len(modified_keys)}, 删除: {len(deleted_keys)})，开始增量同步..."
        )

        # 清理已删除文件的旧向量
        for rel in deleted_keys:
            await self.delete_by_file_path(rel)

        # 增量更新新增与被修改的文件
        for rel in added_keys | modified_keys:
            abs_p = current_files[rel]["abs_path"]
            await self.index_file(abs_p, base_dir=str(repo_path))

        # 保存更新后的元数据缓存
        try:
            meta_file.write_text(
                json.dumps(current_files, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[RAG 知识库] 保存元数据指纹失败: {e}")

        print(f"[RAG 知识库] ✅ 增量同步完成！")
        return {
            "status": "incrementally_synced",
            "added": len(added_keys),
            "modified": len(modified_keys),
            "deleted": len(deleted_keys),
            "total_files": len(current_files),
        }

    async def hybrid_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        执行全异步双路召回并通过 RRF (Reciprocal Rank Fusion) 自动融合打分
        """
        if not self._collection_initialized:
            await self.setup_collection()

        # 1. 异步计算 Query 的 Dense 向量
        dense_query = await self.dense_model.aembed_query(query)

        # 2. 计算 Query 的 Sparse BM25 向量（FastEmbed 在线程池运行）
        sparse_query_obj = (
            await asyncio.to_thread(lambda: list(self.sparse_model.embed([query])))
        )[0]
        sparse_query = models.SparseVector(
            indices=sparse_query_obj.indices.tolist(),
            values=sparse_query_obj.values.tolist(),
        )

        # 3. 使用 query_points 进行 Prefetch + RRF 原生异步融合检索
        search_result = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # Dense 路：语义召回
                models.Prefetch(
                    query=dense_query, using="dense_vector", limit=top_k * 3
                ),
                # Sparse 路：精确关键词/符号 BM25 召回
                models.Prefetch(
                    query=sparse_query, using="sparse_vector", limit=top_k * 3
                ),
            ],
            # RRF (倒数排名融合)：根据两路召回的排名综合打分
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        # 4. 解析结果
        results = []
        for point in search_result.points:
            results.append({"score": point.score, **(point.payload or {})})
        return results

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """释放 AsyncQdrantClient 连接资源"""
        await self.client.close()
