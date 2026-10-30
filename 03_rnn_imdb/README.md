# 03 IMDb：RNN 文本分类

本模块用双向 LSTM 判断影评是正面还是负面，展示传统 NLP 中从原始文本到序列分类的完整链路。

## 运行

```bash
python 03_rnn_imdb/model.py
python 03_rnn_imdb/train.py
```

首次运行会通过 Hugging Face Datasets 下载 IMDb。产物包括 `imdb_lstm.pt` 和 `imdb_vocab.json`。

## 数据和模型流程

```text
影评字符串
-> str.split() 空白分词
-> 训练集词频表
-> token ID（<pad>=0, <unk>=1）
-> 截断/补齐到 200
-> Embedding(25000,128)
-> 两层双向 LSTM，每个方向 hidden=256
-> 拼接最后一层正向/反向状态
-> Dropout + Linear(512,2)
-> 正面/负面 logits
```

这里的 Embedding 是模型内部可训练的词向量表，与 RAG 中用于语义检索的句向量模型职责不同。

## 重要知识

### token ID 没有大小关系

词 ID 只是索引，ID 100 并不比 ID 10 “更有语义”。Embedding 表将离散 ID 映射为可学习的稠密向量，语义关系来自训练，而不是编号。

### 双向 LSTM 不适合自回归生成

双向 LSTM 同时读取左右上下文，适合整句分类。生成下一个 token 时未来内容不存在，因此自回归语言模型不能直接使用未来方向。

### `padding_idx=0` 不代表 LSTM 忽略 padding

它只让 pad embedding 固定为零。LSTM 仍会在补齐的时间步继续递归更新隐藏状态。若要真正跳过 pad，应传递真实长度并使用 `pack_padded_sequence`，或对所有时间步输出做 masked pooling。

### 预处理也是模型的一部分

部署不只需要权重和词表，还需要：分词规则、大小写规则、最大长度、特殊 token ID、标签映射和模型结构。任何一项变化都可能让同一权重产生不同结果。

## 容易忽略的建模选择

- `str.split()` 不会规范大小写、标点和 IMDb 中的 HTML 标记，同一个词可能产生多种表面形式。
- 当前只保留前 200 个 token，可能丢掉影评结尾的总结性情绪。
- 最终状态只是序列聚合方法之一，也可使用 masked mean/max 或 attention pooling。
- 词表使用 top-k，没有最低词频门槛；稀有词可能占用大量 embedding 参数。
- 每个 epoch 都在 collate 中重新分词和查词表，工程上可以预编码并缓存。

## 教学版边界

- 使用 IMDb test split 的前 2000 条作为验证集，没有保留独立最终测试集。
- 固定取前 2000 条，没有检查类别是否平衡。
- 注释提到 RNN 梯度爆炸，但代码尚未执行梯度裁剪。
- 只改 `bidirectional=False` 会使状态提取和分类头维度不匹配，必须同步修改。
- 保存最终 epoch，而不是验证最优模型。
- 推理演示仍使用内存中的词表，没有验证从 JSON 独立恢复完整流程。

## 建议实验

1. 实现 packed sequence，比较短文本上的准确率和训练速度。
2. 比较前 200、后 200、前后各 100 个 token。
3. 加入 lowercase、HTML 清理与标点处理，统计 `<unk>` 比例。
4. 比较最终状态、masked mean pooling 和 attention pooling。
5. 记录梯度范数并测试 `clip_grad_norm_`。
6. 从 train 划分 validation，保留完整 test 做最终评估。
7. 比较随机 Embedding 与预训练 GloVe/fastText。

## 与后续模块的联系

LSTM 按时间顺序递归，长距离依赖和并行效率有限。下一模块的 Transformer 用注意力让任意位置直接交互，并允许训练时并行处理整段序列。
