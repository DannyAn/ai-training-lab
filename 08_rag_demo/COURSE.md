# 08 RAG 讲师脚本

建议时长：85 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能区分 Embedding 模型、Chroma 向量数据库和生成 LLM，能画出 ingest/query 数据流，并说明何时必须重建索引。

## 开场问题

提问：“公司制度每天变化，应把每次变化都重新微调进 Qwen，还是让模型回答前读取最新制度？”

引出外部知识、可更新性和引用需求。

## 板书

```text
离线：document -> split -> Embedding -> vectors -> Chroma

在线：question -> 同一 Embedding -> query vector
                    -> Chroma top-k -> source chunks
question + chunks -> LLM -> answer
```

用三种颜色标出：模型、数据库、数据。

## 三个组件的口播

- Embedding 模型是文本编码器，输出数字向量，不保存文档、不生成答案。
- Chroma 是向量数据库，保存向量、原文和 metadata，不是大模型。
- Qwen 是生成 LLM，只在 `setup_llm()` 后加载，读取检索原文组织答案。

强调：`chroma_db` 是本地目录名，ChromaDB 是数据库产品名。

## 代码走读

1. `train.py` 的 `EMBED_MODEL`、`LLM_MODEL`、`DB_DIR`。
2. `model.py` 初始化：Embedding 和已有 Chroma。
3. `ingest()`：read text、Document、split、delete collection、from_documents。
4. `setup_llm()`：HuggingFacePipeline、retriever、LCEL Runnable。
5. `query()`：result 与 source metadata。
6. 回到 `train.py`：先纯 retrieval，完整生成默认被注释。

## 现场演示

```bash
python 08_rag_demo/train.py
```

演示顺序必须是：

1. 观察示例文档。
2. 观察 chunk 数量。
3. 提问“什么是监督学习”，只看检索原文。
4. 提问库外问题“什么是量子计算”，观察 top-k 仍返回内容。
5. 最后才选择是否加载 Qwen 生成答案。

## 必讲误区

- RAG 不是把文档训练进 Qwen。
- 相同向量维度不代表相同语义空间；换 Embedding 必须重建库。
- 换生成 LLM 通常不用重建库。
- top-k 是最接近的 k 条，不代表真的相关。
- 当前 ingest 删除旧 collection 全量重建，不是生产增量更新。
- ChromaDB 不是玩具，但当前进程内本地目录模式是教学/单机原型配置。
- 检索文档是不可信输入，可能包含间接 prompt injection。

## 课堂实验

把 chunk size 改为 100，构造“监督学习和强化学习有什么区别”的问题，比较 top-1 与 top-3。要求先只评价正确 chunk 是否召回，再讨论 LLM 回答。

进阶：为库外问题增加 similarity score threshold，无可靠结果时不调用 LLM。

## 提问

1. `chroma_db` 中是否保存 Qwen 权重？
2. 为什么更换 Embedding 后必须重新 ingest？
3. 更换 Qwen 为 DeepSeek distilled 模型是否要重建向量库？
4. 检索准确但回答错误，与检索错误但回答流畅，应分别如何定位？
5. 本地 Chroma 与生产数据库服务还差哪些能力？

## 收束

用最终决策表结束课程：事实更新与引用选 RAG；输出行为与格式适配选 LoRA；分类任务可选专用模型微调；复杂系统可以组合，而不是寻找一个万能模型。
