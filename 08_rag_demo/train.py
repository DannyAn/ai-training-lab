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

# 一个完整的 RAG 通常包含三个组件，并不是把知识“训练进”某个大模型：
# 1. Embedding 模型：一个文本编码器。输入一段文本，输出一串固定维度的数字
#    （本例为 384 维向量）。它在 ingest 时编码文档，在检索时编码用户问题。
# 2. Chroma 向量数据库：保存“文档向量 + 文档原文”，并查找最接近问题向量的文档。
#    Chroma 是数据库，不是模型；Embedding 模型也不是数据库类型。
# 3. 生成模型（LLM）：读取“用户问题 + Chroma 找到的文档原文”，生成最终答案。
#
# 数据流：
#   建库：文档原文 -> Embedding 模型 -> 文档向量 -> 存入 Chroma
#   问答：问题 -> 同一个 Embedding 模型 -> 问题向量 -> Chroma 检索相关原文
#         -> 把“问题 + 相关原文”交给 LLM -> 生成答案
#
# 文档和问题必须使用同一个 Embedding 模型编码，向量才处于同一个语义空间。
# 因此更换 EMBED_MODEL 后必须重新执行 ingest() 建库；更换 LLM_MODEL 则无需重建库。
#
# 当前默认示例使用本地 Hugging Face 模型，不会调用 Qwen、DeepSeek 等厂商的在线 API。
# 主流开源 Embedding模型（按场景）： 
# BEG系列: 中文/中英文混合（无脑首选），bge-large-zh/m3
# E5系列：通用资源紧张/边缘设备, e5-small-v2/e5-large-v2
# GTE系列： 多语言, multilingual-22-12, gte-base, gte-large
# Jina v2：常温（>=8kK token), jina-embedding-v2, jina-embedding-v2-large
# OpenAI ada/3: 省事，不想本地部署, text-embedding-ada-002, text-embedding-3
# 口诀：中文无脑选 BGE；多语言 GTE，省钱省事 OpenAI; 长文档 Jina v2；边缘设备 E5 - small。
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def create_sample_doc():
    """创建示例文档用于 RAG 测试"""
    doc_path = "./08_rag_demo/data/ml_intro.txt"
    if os.path.exists(doc_path):
        print(f"示例文档已存在，跳过创建: {doc_path}")
        return
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
    with open(doc_path, "w") as f:
        f.write(content)
    print(f"示例文档已创建: {doc_path}")


def main():
    print("=== RAG Demo ===\n")

    # 【记忆】RAG 全流程(见 model.py RAGSystem)：
    #   离线建库：文档 -> 切分 -> Embedding -> 存入向量库            (一次性，ingest())
    #   在线问答：问题 -> Embedding -> 向量库检索 top-k -> 拼进 prompt -> LLM 生成  (query())
    #   本质：给 LLM 外挂一个"可检索的私有记忆"，无需重新训练就能答私有知识。
    create_sample_doc()

    # 【记忆】这里配置的不是“给 Qwen 做一个 RAG”，而是配置 RAG 的三个独立组件：
    #   EMBED_MODEL：负责把文本编码成向量。
    #   DB_DIR/Chroma：负责保存向量和原文，并执行相似度搜索。
    #   LLM_MODEL：只负责利用检索出的原文组织答案。
    rag = RAGSystem(
        persist_dir=DB_DIR,
        embed_model=EMBED_MODEL,
        llm_model=LLM_MODEL,
    )
    print(f"Embedding 模型: {EMBED_MODEL}")
    print(f"生成模型: {LLM_MODEL} (仅在调用 setup_llm() 后加载)")

    # 【调参】ingest() 会读取原文、切分文本、调用 EMBED_MODEL 计算每块的向量，
    #   最后把向量和原文一起存入 Chroma。它不会把文档喂给 Qwen，也不会训练 Qwen。
    #   切分参数(在 model.py ingest 里)：
    #   chunk_size：每块多少字符。太大->检索不精确、塞爆 LLM 上下文；太小->语义不完整。
    #   chunk_overlap：相邻块重叠字数，防止把一句话切断丢上下文。常用 50~100。
    rag.ingest(["./08_rag_demo/data/ml_intro.txt"])

    # 【产物】产物是 Chroma 向量数据库(chroma_db/ 目录)，存了每个文档块的向量 + 原文。
    #   数据库中没有 Qwen/DeepSeek 的模型权重，也没有生成好的答案。
    #   下次启动 RAGSystem 会自动加载(见 model.py 的 os.path.exists 分支)，无需重新 ingest。
    print(f"向量库已持久化到: {DB_DIR}")

    # 【使用】检索演示：不依赖 LLM，只看"能不能找回相关片段"。
    #   这一步能独立验证知识库质量--检索不准的话，后面 LLM 答得再好也没用。
    # retriever = rag.db.as_retriever(search_kwargs={"k": 2})  # 【调参】k：取几条最相关片段
    # docs = retriever.invoke("什么是监督学习？")
    # print(f"\n检索结果 ({len(docs)} chunks):")
    # for i, doc in enumerate(docs, 1):
    #     print(f"\n--- Chunk {i} ---")
    #     print(doc.page_content[:200] + "...")

    # 【使用】完整问答(需 LLM，本机较重，按需取消下面三行的注释)：
    rag.setup_llm()                         # 下载并加载上面配置的 Qwen2.5-1.5B
    result = rag.query("什么是深度学习？")   # MiniLM 检索，再由 Qwen 根据片段生成答案
    print(result["result"], result["sources"])
    
    # query() 不是在“查找 Qwen 或 DeepSeek”：
    #   1. EMBED_MODEL 把问题向量化，并从 Chroma 中检索相关文档片段。
    #   2. LLM_MODEL 接收问题和片段，生成自然语言答案。
    #   3. 当前 LLM_MODEL 明确是 Qwen，因此不会调用 DeepSeek。
    #
    # 可运行的本地 DeepSeek 替换示例（修改上面的 LLM_MODEL 后再运行）：
    #   LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    # 该模型可以被当前 HuggingFacePipeline 下载到本机运行。DeepSeek V4Pro 等托管
    # API 模型不能只改模型名称接入；它们需要对应服务商的 LangChain ChatModel 适配器、
    # API Key 和 endpoint，而不是本示例使用的本地 HuggingFacePipeline。
    print("\n提示: 取消注释，启用完整问答: rag.setup_llm(); rag.query('什么是深度学习？')")

    # 【练习】
    #   1. chunk_size 从 500 调到 100，问"监督学习和强化学习的区别"，看检索能否同时命中两段。
    #   2. 问一个文档里没有的问题(如"什么是量子计算")，观察检索回的片段其实不相关(RAG 的局限)。
    #   3. 换个 Embedding 模型(如 all-MiniLM-L12-v2)，看检索质量变化。


if __name__ == "__main__":
    main()
