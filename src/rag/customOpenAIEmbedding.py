import os
import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import List, Optional


class CustomOpenAIEmbedding:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        max_retries: int = 3,
    ):
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.model = model or os.getenv("EMBEDDING_MODEL")
        self.max_retries = max_retries
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=10 * 60,
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=10 * 60,
        )
        self._dim = dimension

    async def _async_create_with_retry(self, input_texts: List[str]):
        """带重试机制的异步 Embedding 请求（应对冷启动/504等瞬时异常）"""
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return await self.async_client.embeddings.create(
                    input=input_texts, model=self.model
                )
            except Exception as e:
                last_exception = e
                wait_time = (attempt + 1) * 2
                print(
                    f"[Embedding 请求重试 {attempt + 1}/{self.max_retries}] 异常: {e}, 等待 {wait_time}s 后重试..."
                )
                await asyncio.sleep(wait_time)
        raise last_exception

    async def aembed_documents(
        self, texts: List[str], batch_size: int = 1, max_concurrency: int = 3
    ) -> List[List[float]]:
        """
        异步并发/分批计算文档嵌入向量
        - 当 batch_size=1 且启用 max_concurrency 时，采用异步信号量并发单条请求，规避大 payload 网关超时
        """
        if not texts:
            return []

        if batch_size == 1 and len(texts) > 1:
            sem = asyncio.Semaphore(max_concurrency)

            async def _embed_one(idx: int, text: str):
                async with sem:
                    resp = await self._async_create_with_retry([text])
                    return idx, resp.data[0].embedding

            tasks = [_embed_one(i, t) for i, t in enumerate(texts)]
            indexed_embeddings = await asyncio.gather(*tasks)
            indexed_embeddings.sort(key=lambda x: x[0])
            all_embeddings = [emb for _, emb in indexed_embeddings]
        else:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self._async_create_with_retry(batch)
                sorted_data = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([item.embedding for item in sorted_data])

        if not self._dim and all_embeddings:
            self._dim = len(all_embeddings[0])
        return all_embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """异步计算单条查询语句的嵌入向量"""
        response = await self._async_create_with_retry([text])
        embedding = response.data[0].embedding
        if not self._dim:
            self._dim = len(embedding)
        return embedding

    async def aget_dimension(self) -> int:
        """异步获取向量维度（若未指定则通过查询探测）"""
        if self._dim is None:
            await self.aembed_query("probe")
        return self._dim

    def embed_documents(
        self, texts: List[str], batch_size: int = 64
    ) -> List[List[float]]:
        """同步批量计算文档嵌入向量（保留兼容）"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([item.embedding for item in sorted_data])

        if not self._dim and all_embeddings:
            self._dim = len(all_embeddings[0])
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """同步计算单条查询语句的嵌入向量（保留兼容）"""
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        """获取向量维度（同步属性）"""
        if self._dim is None:
            self._dim = len(self.embed_query("probe"))
        return self._dim
