# 04 从零实现 Transformer

本模块实现 encoder-decoder Transformer，用于理解注意力、位置编码、mask、teacher forcing 和自回归解码。它展示的是结构和训练机制，不是一个真实翻译系统。

## 运行

```bash
python 04_transformer_from_scratch/model.py
python 04_transformer_from_scratch/train.py
```

训练产物是 `04_transformer_from_scratch/transformer_demo.pt`。

## 架构

```text
source IDs -> Embedding + 位置编码 -> Encoder x 3 -> memory
target IDs -> Embedding + 位置编码 -> Decoder x 3 -> 词表 logits
```

每个 Encoder 层包含 self-attention 和 FFN；每个 Decoder 层包含 masked self-attention、对 encoder memory 的 cross-attention 和 FFN。

训练脚本使用 `d_model=128`、4 个 head、3 层和 `d_ff=512`。每个 head 的维度是 32。

## 三种注意力不要混淆

- Encoder self-attention：source 内部互相查看。
- Decoder causal self-attention：target 只能查看当前位置及之前的位置。
- Cross-attention：decoder query 查看 encoder 输出，建立 source 到 target 的条件关系。

因果 mask 只解决“不能偷看未来”，不解决 padding。padding mask 是另一类 mask。

## Teacher forcing

训练时使用：

```text
tgt_in  = target[:-1]
tgt_out = target[1:]
```

模型一次并行预测所有下一 token；推理时没有真实目标，只能把自己刚生成的 token 追加回输入，逐步生成。这种训练/推理输入差异也称 exposure bias。

## 重要隐藏知识

### 当前实现是 Pre-LN

代码执行 `x + Sublayer(LayerNorm(x))`，属于 Pre-LayerNorm。原始 Transformer 论文使用的经典排列更接近 Post-LN。Pre-LN 通常更容易训练深层网络；规范实现常在整个堆叠末尾再加 final LayerNorm，当前代码没有。

### head 数变化不等于参数量成比例变化

固定 `d_model` 时，Q/K/V/O 都仍是 `d_model -> d_model` 投影。把 4 个 head 改为 1 个主要改变每头维度和注意力分工，线性投影参数量基本不变。

### `ignore_index=0` 不会阻止注意力读取 pad

它只让目标 pad 不计入 loss。当前模型没有正确传递 source/target padding mask，而且 Embedding 也未设置 `padding_idx=0`，所以 pad 仍会进入 encoder 和 cross-attention。

### 位置编码也是 checkpoint 状态

正弦位置编码不是可训练参数，但通过 buffer 保存。默认 `max_len=5000` 远大于 toy 序列长度，因此会给 checkpoint 增加可观体积。

## Toy 数据的真实含义

当前 source 和 target 都是独立随机序列，没有翻译关系。有效目标 token 有 48 种，无法学习规律时理论交叉熵约为：

```text
ln(48) ≈ 3.87
```

因此该脚本只能验证张量形状、mask、loss、反向传播和生成循环，不能证明模型学会翻译。固定随机样本上的 loss 下降也可能只是记忆训练集。

## 教学版边界

- target 中没有真正的 BOS/EOS；推理使用的 BOS=1 在训练 target 中从未出现。
- 没有 EOS，所以生成只能固定执行 12 步，不能自然停止。
- source padding 未屏蔽。
- 推理每一步都重新计算 encoder 和全部 decoder，没有 encoder cache 或 KV cache。
- cross-attention 使用同一个 LayerNorm 处理 decoder 和 encoder memory，不是常见标准实现。
- 没有验证集、学习率 warmup、配置文件和 tokenizer。
- 奇数 `d_model` 可能让正弦位置编码维度不匹配。

## 建议实验

1. 将数据改为可学习的复制、反转或 token 加一任务。
2. 加入 BOS、EOS，并让生成遇到 EOS 停止。
3. 实现 source 和 target padding mask，观察 pad attention 权重。
4. 去掉 causal mask，比较训练 loss 与真实自回归生成，观察信息泄漏。
5. 比较 Pre-LN、Pre-LN + final norm、Post-LN。
6. 比较 1、2、4、8 个 head 的参数量、速度和效果。
7. 比较无位置编码、正弦位置编码和可学习位置 Embedding。
8. 缓存 encoder memory，再进一步实现 decoder KV cache，测量生成长度增长时的耗时。

## 与 GPT 的关系

本模块是 encoder-decoder 架构，适合“输入序列 -> 输出序列”。下一模块 GPT Tiny 是 decoder-only 架构，只保留带因果 mask 的 Transformer block，用统一的 next-token prediction 目标进行语言建模。
