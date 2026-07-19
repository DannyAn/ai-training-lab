"""RAG 检索增强生成演示

学习目标：理解 RAG 流程--这不是"训练模型"，而是"用现成模型 + 私有知识库做问答"。
注意：本模块没有训练循环。产物是"向量数据库"，不是权重文件。
注释标记(适配 RAG)：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
import os
sys.path.append(".")

from model import RAGSystem

DB_DIR = "./08_rag_demo/chroma_db"


def create_sample_doc():
    """创建示例文档用于 RAG 测试"""
    content = """
机器学习是人工智能的核心领域之一。它使计算机能够从数据中学习模式和规律，
而无需进行明确编程。主要分为监督学习、无监督学习和强化学习三大类。

监督学习使用带标签的数据进行训练，常见任务包括分类和回归。
无监督学习则从未标记的数据中发现隐藏结构，如聚类和降维。
强化学习通过与环境交互并获得奖励信号来学习最优策略。

深度学习是机器学习的一个子领域，使用多层神经网络来学习数据的层次化表示。
它在图像识别、自然语言处理、语音识别等领域取得了突破性进展。

迁移学习允许将在一个任务上学到的知识应用到相关任务中，
大大减少了训练数据和计算资源的需求。
"""
    os.makedirs("./08_rag_demo/data", exist_ok=True)
    with open("./08_rag_demo/data/ml_intro.txt", "w") as f:
        f.write(content)
    print("示例文档已创建: ./08_rag_demo/data/ml_intro.txt")


def main():
    print("=== RAG Demo ===\n")

    # 【记忆】RAG 全流程(见 model.py RAGSystem)：
    #   离线建库：文档 -> 切分 -> Embedding -> 存入向量库            (一次性，ingest())
    #   在线问答：问题 -> Embedding -> 向量库检索 top-k -> 拼进 prompt -> LLM 生成  (query())
    #   本质：给 LLM 外挂一个"可检索的私有记忆"，无需重新训练就能答私有知识。
    create_sample_doc()

    # 【记忆】Embedding 模型把文本变成向量(如 384 维)，语义相近的文本向量也相近。
    #   检索就是用"问题向量"找"文档向量"里最近的 k 个 -> 语义检索(比关键词匹配强)。
    rag = RAGSystem(persist_dir=DB_DIR)

    # 【调参】切分参数(在 model.py ingest 里)：
    #   chunk_size：每块多少字符。太大->检索不精确、塞爆 LLM 上下文；太小->语义不完整。
    #   chunk_overlap：相邻块重叠字数，防止把一句话切断丢上下文。常用 50~100。
    rag.ingest(["./08_rag_demo/data/ml_intro.txt"])

    # 【产物】产物是向量数据库(chroma_db/ 目录)，存了每个文档块的向量 + 原文。
    #   下次启动 RAGSystem 会自动加载(见 model.py 的 os.path.exists 分支)，无需重新 ingest。
    print(f"向量库已持久化到: {DB_DIR}")

    # 【使用】检索演示：不依赖 LLM，只看"能不能找回相关片段"。
    #   这一步能独立验证知识库质量--检索不准的话，后面 LLM 答得再好也没用。
    retriever = rag.db.as_retriever(search_kwargs={"k": 2})  # 【调参】k：取几条最相关片段
    docs = retriever.invoke("什么是监督学习？")
    print(f"\n检索结果 ({len(docs)} chunks):")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:200] + "...")

    # 【使用】完整问答(需 LLM，本机较重，按需启用)：
    #   rag.setup_llm()                              # 加载本地 LLM(会下载 Qwen)
    #   result = rag.query("什么是深度学习？")        # 检索 + LLM 生成
    #   print(result["result"], result["sources"])
    print("\n提示: 启用完整问答: rag.setup_llm(); rag.query('什么是深度学习？')")

    # 【练习】
    #   1. chunk_size 从 500 调到 100，问"监督学习和强化学习的区别"，看检索能否同时命中两段。
    #   2. 问一个文档里没有的问题(如"什么是量子计算")，观察检索回的片段其实不相关(RAG 的局限)。
    #   3. 换个 Embedding 模型(如 all-MiniLM-L12-v2)，看检索质量变化。


if __name__ == "__main__":
    main()
