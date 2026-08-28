import asyncio
from dotenv import load_dotenv

load_dotenv()

from qdrantCodeHybridRAG import QdrantCodeHybridRAG
from codebaseLoader import CodebaseLoader

# 1. 初始化 RAG 实例
rag = QdrantCodeHybridRAG(
    collection_name="my_project_kb",
    qdrant_path="C:/Users/zengd/Desktop/lc_agent/qdrant_hybrid_storage",
)


async def main():
    print("🔧 1. 初始化 Qdrant 集合与 Schema...")
    await rag.setup_collection()

    print("\n🔍 2. 正在扫描并切分代码库...")
    # 2. 全异步批量并发计算 Embedding 并写入 Qdrant (原生 AsyncQdrantClient + AsyncOpenAI)
    await rag.index_directory(
        repo_dir="C:/Users/zengd/Desktop/lc_agent/src", batch_size=1
    )

    # 3. 执行全异步测试检索验证
    print("\n🔎 执行全异步测试检索: '获取 agent 实例'")
    results = await rag.hybrid_search("获取 agent 实例", top_k=2)
    for idx, res in enumerate(results, 1):
        print(
            f"\nTop {idx} [Score: {res['score']:.4f}] 文件: {res.get('file_path')} | 符号: {res.get('name')}"
        )
        print(f"预览:\n{res.get('content', '')[:150]}...")

    # 4. 关闭连接释放资源
    await rag.close()
    print("\n🎉 全异步流程测试完毕！")


if __name__ == "__main__":
    asyncio.run(main())
