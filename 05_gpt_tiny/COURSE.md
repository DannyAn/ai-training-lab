# 05 GPT Tiny 讲师脚本

建议时长：80 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能解释 next-token prediction、字符级 tokenizer、因果注意力和自回归采样，并理解 perplexity 依赖 token 粒度。

## 开场问题

写出 `Hello worl`，让学员预测下一个字符。追问：“模型是在背一句话，还是学习一个条件概率分布？”

## 板书

```text
x: H e l l o
y: e l l o _

P(x_t | x_0 ... x_{t-1})
loss = 所有位置 next-token cross entropy
perplexity = exp(loss)
```

画出 decoder-only block，并与上一节 encoder-decoder 对照。

## 代码走读

1. `CharDataset`：词表、滑动窗口和一位右移标签。
2. `model.py`：token + position embedding。
3. 融合 QKV 投影和 causal mask buffer。
4. Pre-LN GPT block、final norm 和 lm_head。
5. 展平 `(B,T,V)` 与 `(B,T)` 计算 loss。
6. `generate()`：最后位置 logits、temperature、top-k、multinomial。

## 现场演示

```bash
python 05_gpt_tiny/model.py
```

如果已有 checkpoint，展示同一 prompt 在不同 temperature/top-k 下的多次输出。不要只展示一个样本后下质量结论。

## 必讲误区

- 字符级 perplexity 不能与 BPE 模型直接比较。
- 高度重叠窗口不是独立样本，随机拆窗口会泄漏。
- 三个短句重复语料主要测试记忆，不代表语言理解。
- temperature=0 不是当前代码里的贪心，会发生除零；贪心应显式 argmax。
- 位置 Embedding 大小才是真实上下文上限，mask 更大不能扩容。
- 没有 KV cache 时，每步重算全部上下文。

## 课堂实验

使用同一 checkpoint，以 temperature `0.2/0.8/1.2` 和 top-k `5/10/全部` 各生成三次。记录重复度、合法词比例和多样性，说明采样参数改变的是输出分布，不是模型知识。

## 提问

1. 为什么训练时可以一次预测 64 个位置？
2. 因果 mask 去掉后训练 loss 为什么可能更好，生成反而失败？
3. 词表外字符会发生什么？
4. KV cache 缓存的是哪些量？

## 收束

Tiny GPT 从零学习，需要大量数据和计算。下一节使用已经预训练好的 BERT，只用较小下游数据适配任务。
