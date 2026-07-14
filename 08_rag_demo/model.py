"""RAG 检索增强生成系统

流程: 文档 → 切分 → Embedding → 向量存储 → 检索 → LLM 生成
"""

import os
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


class RAGSystem:
    """简单的 RAG 问答系统"""

    def __init__(
        self,
        persist_dir: str = "./08_rag_demo/chroma_db",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # 如果已有向量库则加载，否则为 None
        if os.path.exists(persist_dir):
            self.db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
        else:
            self.db = None

        self.llm = None
        self.qa_chain = None

    def ingest(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
        """导入文档到向量数据库"""
        docs = []
        for fp in file_paths:
            loader = TextLoader(fp)
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        self.db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_dir
        )
        self.db.persist()
        print(f"Ingested {len(chunks)} chunks from {len(file_paths)} file(s)")

    def setup_llm(self):
        """初始化本地 LLM 用于问答"""
        pipe = pipeline(
            "text-generation",
            model=self.llm_model if "Qwen" in self.llm_model else None,
            model_kwargs={"device_map": "auto"},
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, retriever=retriever, return_source_documents=True
        )

    def query(self, question: str) -> dict:
        """回答用户问题"""
        if self.qa_chain is None:
            return {"query": question, "result": "请先调用 setup_llm() 和 ingest() 初始化系统"}

        result = self.qa_chain.invoke({"query": question})
        return {
            "query": question,
            "result": result["result"],
            "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]],
        }


if __name__ == "__main__":
    print("RAGSystem 初始化完成")
    print()
    print("使用示例:")
    print("  rag = RAGSystem()")
    print("  rag.ingest(['document.txt'])")
    print("  rag.setup_llm()")
    print("  result = rag.query('你的问题')")
