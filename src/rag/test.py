import asyncio, os, sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from dotenv import load_dotenv

load_dotenv()

from rag.qdrantCodeHybridRAG import QdrantCodeHybridRAG
from rag.codeContextRetriever import CodeContextRetriever


async def index():
    async with QdrantCodeHybridRAG(
        collection_name="snake_kb",
        qdrant_path=os.environ["QDRANT_PATH"],
    ) as rag:
        print("\n🔍 1. 正在扫描并切分代码库...")
        # 1. 全异步批量并发计算 Embedding 并写入 Qdrant (原生 AsyncQdrantClient + AsyncOpenAI)
        await rag.index_directory(repo_dir="C:/Users/zengd/Desktop/Snake", batch_size=1)

        # 2. 执行全异步测试检索验证
        print("\n🔎 2. 执行全异步测试检索: '随机生成食物'")
        results = await rag.hybrid_search("随机生成食物", top_k=2)
        for idx, res in enumerate(results, 1):
            print(
                f"\nTop {idx} [Score: {res['score']:.4f}] 文件: {res.get('file_path')} | 符号: {res.get('name')}"
            )
            print(f"预览:\n{res.get('content', '')[:150]}...")

    print("\n🎉 3. 全异步流程测试完毕（连接已自动释放）！")


async def retriever():
    async with QdrantCodeHybridRAG(
        collection_name="snake_kb",
        qdrant_path=os.environ["QDRANT_PATH"],
    ) as rag:
        retriever_instance = CodeContextRetriever(rag=rag)
        prompt = await retriever_instance.retrieve_context_for_query("随机生成食物")
        print("\n=== 检索并构建的 Prompt ===")
        print(prompt)


if __name__ == "__main__":
    asyncio.run(retriever())
