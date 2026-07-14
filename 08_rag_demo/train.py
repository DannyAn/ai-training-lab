"""RAG 系统使用演示"""

import sys
import os
sys.path.append(".")

from model import RAGSystem


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
    print("Sample document created at ./08_rag_demo/data/ml_intro.txt")


def main():
    print("=== RAG Demo ===\n")

    # 创建示例文档
    create_sample_doc()

    # 初始化 RAG 系统
    rag = RAGSystem(persist_dir="./08_rag_demo/chroma_db")

    # 导入文档
    rag.ingest(["./08_rag_demo/data/ml_intro.txt"])

    # 设置 LLM (如果 HuggingFace 资源有限，可跳过或在外部设置)
    # rag.setup_llm()

    # 演示检索能力 (使用稀疏检索，不依赖 LLM)
    retriever = rag.db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("什么是监督学习？")
    print(f"\n检索结果 ({len(docs)} chunks):")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content[:200] + "...")

    print("\n\n提示: 要启用完整问答功能，请运行:")
    print("  rag.setup_llm()")
    print("  result = rag.query('什么是深度学习？')")


if __name__ == "__main__":
    main()
