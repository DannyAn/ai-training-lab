"""RAG 检索增强生成系统

流程: 文档 → 切分 → Embedding → 向量存储 → 检索 → LLM 生成
"""

import os
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter


RAG_PROMPT = PromptTemplate.from_template(
    """请仅根据以下上下文回答问题。不知道时请明确回答不知道。

上下文：
{context}

问题：{question}

回答："""
)


def format_docs(docs) -> str:
    """把检索到的文档块拼接成 LLM 上下文。"""
    return "\n\n".join(doc.page_content for doc in docs)


class RAGSystem:
    """组合文本编码器、Chroma 向量数据库和本地 LLM 的 RAG 系统。"""

    def __init__(
        self,
        persist_dir: str = "./08_rag_demo/chroma_db",
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        """配置向量库、Embedding 模型和生成模型。

        embed_model 是文本编码模型，负责 ingest 时的文档向量化以及检索时的
        问题向量化；persist_dir 指向保存向量和原文的 Chroma 数据库；llm_model
        只在 setup_llm 后用于根据检索出的原文生成答案。Embedding 模型不是
        数据库类型，更换它后需要重新 ingest，确保文档和问题处于同一向量空间。
        """
        self.persist_dir = persist_dir
        self.llm_model = llm_model
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
            docs.append(
                Document(
                    page_content=Path(fp).read_text(encoding="utf-8"),
                    metadata={"source": fp},
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        if self.db is not None:
            self.db.delete_collection()
        self.db = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=self.persist_dir
        )
        print(f"Ingested {len(chunks)} chunks from {len(file_paths)} file(s)")

    def setup_llm(self):
        """初始化本地 LLM 用于问答"""
        if self.db is None:
            raise RuntimeError("请先调用 ingest() 创建向量库")

        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.llm_model,
            task="text-generation",
            device_map="auto",
            pipeline_kwargs={
                "max_new_tokens": 256,
                "do_sample": True,
                "temperature": 0.3,
                "return_full_text": False,
            },
        )

        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        answer_chain = (
            lambda inputs: {
                "context": format_docs(inputs["documents"]),
                "question": inputs["question"],
            }
        ) | RAG_PROMPT | self.llm | StrOutputParser()
        self.qa_chain = RunnableParallel(
            documents=retriever,
            question=RunnablePassthrough(),
        ).assign(result=answer_chain)

    def query(self, question: str) -> dict:
        """回答用户问题"""
        if self.qa_chain is None:
            return {"query": question, "result": "请先调用 setup_llm() 和 ingest() 初始化系统"}

        response = self.qa_chain.invoke(question)
        return {
            "query": question,
            "result": response["result"],
            "sources": [
                doc.metadata.get("source", "unknown")
                for doc in response["documents"]
            ],
        }


if __name__ == "__main__":
    print("RAGSystem 初始化完成")
    print()
    print("使用示例:")
    print("  rag = RAGSystem()")
    print("  rag.ingest(['document.txt'])")
    print("  rag.setup_llm()")
    print("  result = rag.query('你的问题')")
