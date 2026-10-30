# 04 Transformer 讲师脚本

建议时长：90 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能区分 encoder self-attention、decoder causal self-attention 和 cross-attention，能解释 causal mask 与 padding mask不是一回事。

## 开场问题

写一句“动物没有过马路，因为它太累了”，提问：“理解‘它’需要跨越多远？RNN 要走几步，attention 的路径有多长？”

## 板书

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

Encoder: source self-attention
Decoder: masked target self-attention + cross-attention(source memory)
```

画 4x4 下三角 causal mask，并单独画 pad 列被屏蔽的 padding mask。

## 代码走读

1. `MultiHeadAttention`：Q/K/V 投影、拆 head、scores、mask、拼接。
2. `PositionalEncoding`：为什么 attention 自身不知道顺序。
3. Encoder/Decoder layer：指出当前是 Pre-LN。
4. `train.py`：动态 padding、`tgt[:, :-1]` 与 `tgt[:, 1:]`。
5. `ignore_index=0`：只影响 loss，不会自动影响 attention。
6. greedy decode：训练并行、推理逐步，以及重复计算。

## 现场演示

```bash
python 04_transformer_from_scratch/model.py
```

现场打印一个长度 5 的 causal mask，让学员逐行解释哪些位置允许查看。完整 toy 训练可展示预先日志，不把随机 token 输出解释成翻译结果。

## 必讲误区

- 当前 source 和 target 独立随机，没有可学习翻译映射。
- loss 下降可能是记忆固定随机样本，不证明学会语言。
- 当前没有正确 source padding mask。
- `ignore_index` 与 attention mask 职责不同。
- target 缺少真正 BOS/EOS，推理起始 token 是训练分布外输入。
- 固定 d_model 时增加 head 数不会让投影参数量同比增加。

## 课堂实验

把 toy 任务改为“target 是 source 的反转”，加入 BOS/EOS，建立 held-out validation。比较有无 causal mask 的训练 loss 和真实生成结果。

若时间有限，只要求学员设计正确的数据样例和 mask shape，不必课堂完成训练。

## 提问

1. 为什么 scores 要除以 `sqrt(d_k)`？
2. causal mask 和 padding mask 分别防止什么？
3. 为什么训练能并行预测 target，生成却要逐 token？
4. 为什么没有位置编码时 attention 难以区分词序？

## 收束

Encoder-decoder 适合条件序列生成。下一节删除 encoder 和 cross-attention，只保留 causal decoder，得到 GPT 风格语言模型。
