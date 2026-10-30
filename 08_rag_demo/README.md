# 08 RAG：检索增强生成

RAG（Retrieval-Augmented Generation）不是把私有文档重新训练进大模型，而是在回答问题时先检索相关资料，再把资料交给大模型生成答案。

## 运行

只执行建库和检索，不加载生成大模型：

```bash
python 08_rag_demo/train.py
```

默认会下载本地 Embedding 模型、创建示例文档、重建 Chroma 数据库并查询“什么是监督学习？”。完整问答代码在 `train.py` 中按需取消注释，会额外下载 Qwen。

## RAG 的三个组件

一个基础 RAG 通常包含三个独立组件，而不是“两个大模型”：

| 组件 | 本例 | 职责 | 是否保存文档 |
|---|---|---|---:|
| Embedding 模型 | `all-MiniLM-L6-v2` | 将文本编码为 384 维向量 | 否 |
| 向量数据库 | ChromaDB | 保存向量、原文、metadata 并做相似度检索 | 是 |
| 生成模型 LLM | `Qwen2.5-1.5B-Instruct` | 根据问题和检索原文组织答案 | 否 |

Embedding 模型是文本编码器，不是数据库类型；Chroma 才是数据库；Qwen 才是生成答案的大模型。

## 完整数据流

### 离线建库

```text
文档原文
-> RecursiveCharacterTextSplitter 切成 chunk
-> Embedding 模型把每个 chunk 编码成向量
-> Chroma 保存“向量 + chunk 原文 + source metadata”
```

`rag.ingest()` 不会加载、训练或修改 Qwen。

### 在线检索与生成

```text
用户问题
-> 同一个 Embedding 模型生成问题向量
-> Chroma 查找相近的文档向量
-> 返回对应的 chunk 原文
-> Prompt = 用户问题 + 检索原文
-> Qwen 生成最终答案
```

`query()` 不是在“查询 Qwen 或 DeepSeek 的知识库”。它查询的是 Chroma 中的文档；LLM 只是利用检索结果回答。

## 为什么文档和问题必须使用同一个 Embedding 模型

不同 Embedding 模型产生的是不同语义坐标系。即使两个模型都输出 384 维，也不代表同一个维度有相同含义。因此：

- 更换 Embedding 模型后必须重新 ingest。
- 修改 chunk size、overlap 或原文后也应重建索引。
- 只更换 Qwen、DeepSeek 等生成 LLM，通常不需要重建索引。

生产索引应记录 embedding model、revision、维度、归一化设置、chunk 配置和语料版本。当前教学库尚未保存这些指纹。

## ChromaDB 是什么

`chroma_db` 是本例的本地数据目录，ChromaDB 是实际的开源向量数据库产品。当前用法：

```python
Chroma(persist_directory="./08_rag_demo/chroma_db")
```

属于进程内、本地持久化模式，类似应用直接使用本地 SQLite：

- 非常适合教学、Notebook、原型、单机工具和小型知识库。
- Chroma 产品也支持 client/server、自托管服务和 Chroma Cloud。
- 但本仓库的本地目录模式没有生产服务所需的认证、多租户、高可用、备份、并发写入、监控和版本切换。

因此准确结论是：ChromaDB 不是玩具，但这个示例采用的是教学级、单机嵌入式用法。

大型或既有生产系统还常选择 pgvector、Qdrant、Milvus、Weaviate、Elasticsearch/OpenSearch、Pinecone 或云厂商向量服务。选型取决于数据规模、过滤能力、已有基础设施、运维和可用性要求，而不是单纯比较“谁更高级”。

## 当前 LangChain 结构

本模块使用 LangChain 1.x 的拆包集成：

```text
langchain-core             Document、Prompt、Runnable、OutputParser
langchain-text-splitters   文本切分
langchain-huggingface      本地 Embedding 和 HuggingFacePipeline
langchain-chroma           Chroma 向量库适配
```

问答链使用 LCEL `RunnableParallel` 组合检索、Prompt、LLM 和解析器，没有使用旧版 `RetrievalQA`。

## 当前 ingest 的行为

本例每次 ingest 会删除旧 collection 并全量重建，以避免教学脚本重复运行产生重复 chunk。这不适合生产知识库：

- 不是增量 add/update/delete。
- 第二次只 ingest 一个文件会替换之前所有文件。
- 先删除再构建不是原子操作，中途失败会丢失旧索引。

生产方式通常使用稳定 document/chunk ID、内容 hash、增量更新、版本化 collection，并在新索引验证通过后切换。

## 检索质量不能只看最终回答

如果检索错了，再强的 LLM 也只能基于错误上下文回答。应分开评估：

1. Retrieval：正确 chunk 是否进入 top-k，可计算 Recall@k、MRR、nDCG。
2. Generation：答案是否忠于上下文、是否完整、引用是否正确。

当前检索无论问题是否相关都会返回最接近的 chunk。生产系统通常还需要相关度阈值、无结果拒答、MMR、混合关键词检索或 reranker。

## 容易忽略的问题

- 默认 `all-MiniLM-L6-v2` 主要面向英文，而示例文档是中文；真实项目应评估中文或多语种 Embedding。
- `chunk_size=500` 的单位主要是字符，不是 Embedding 或 LLM token。
- metadata 只有文件路径，没有页码、段落、chunk ID 和文档版本，引用不够精确。
- 文档原文直接进入 prompt，可能携带间接 prompt injection，检索内容必须视为不可信数据。
- `top-k` 只表示“最接近的 k 条”，不表示它们真的相关。
- 生成启用了采样，同一问题可能得到不同表述。
- `train.py` 每次仍会主动 ingest；只有单独初始化 `RAGSystem` 且不调用 ingest 才会复用已有库。

## 教学版与生产版差距

生产 RAG 通常还需要：

- PDF/HTML/Office 文档解析与 OCR。
- 文档去重、权限 metadata、增量更新和失败重试。
- Embedding 模型与索引 manifest/version 管理。
- hybrid search、metadata filter、reranker 和离线评测集。
- token budget、上下文去重、精确引用和无答案策略。
- prompt injection 防护、数据权限过滤和敏感信息治理。
- API 服务、认证、多租户、缓存、监控、备份和恢复。

## 建议实验

1. 构造 20 个问题及正确 chunk，先测 Recall@1/3，再启用 LLM。
2. 比较英文 MiniLM、中文和多语种 Embedding；每次更换后重建数据库。
3. 网格测试 chunk size、overlap 和 top-k。
4. 加入相似度阈值，对“量子计算是什么”这类库外问题拒答。
5. 比较普通 similarity 与 MMR，再加入 cross-encoder reranker。
6. 固定 Embedding/Chroma，只替换 Qwen 和可本地运行的 DeepSeek distilled 模型，验证无需重建库。
7. 为 chunk 添加页码、段落、ID 和版本，输出可定位来源。
8. 放入包含恶意指令的文档，测试间接 prompt injection 防护。

## RAG 与微调如何选择

| 需求 | 优先方案 |
|---|---|
| 事实经常更新、需要引用私有文档 | RAG |
| 学习固定输出格式、语气和任务行为 | LoRA/微调 |
| 同时需要私有知识和稳定行为 | RAG + 微调 |

RAG 不是微调的廉价替代，微调也不是数据库。它们解决的是不同问题。
