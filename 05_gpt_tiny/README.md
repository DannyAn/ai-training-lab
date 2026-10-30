# 05 GPT Tiny：字符级自回归语言模型

本模块从零实现一个 decoder-only Transformer，通过预测下一个字符学习文本分布，并展示 temperature 与 top-k 采样。

## 运行

```bash
python 05_gpt_tiny/model.py
python 05_gpt_tiny/train.py
```

产物包括 `gpt_tiny.pt` 和 `char_vocab.json`。

## 数据和训练目标

训练语料由三个短句各重复 1000 次组成。字符集排序后形成字符级词表，每个滑动窗口构造：

```text
x = data[i     : i + 64]
y = data[i + 1 : i + 65]
```

模型在每个位置预测下一个字符。训练时所有位置可以并行计算，但因果 mask 保证位置 `t` 看不到未来答案。

## 模型结构

```text
字符 ID
-> Token Embedding + 可学习位置 Embedding
-> 3 x GPT Block
   -> LayerNorm -> Causal Self-Attention -> 残差
   -> LayerNorm -> 4 倍宽 GELU MLP -> 残差
-> Final LayerNorm
-> Linear 到字符词表
```

训练配置为 `d_model=192`、3 个 head、3 层、上下文长度 64。Q/K/V 使用一个融合的 `Linear(d_model, 3*d_model)` 生成。

## 重要知识

### GPT 与 BERT 的核心差异

GPT 使用因果注意力，只能看左侧上下文，因此可以逐 token 生成。BERT 使用双向注意力，可以同时利用左右上下文，更适合编码和分类，但不能直接按同样方式自回归生成。

### Perplexity 与 tokenizer 有关

本例的 perplexity 单位是字符。字符级、byte 级、BPE 和 WordPiece 的 token 粒度不同，因此 perplexity 数值不能脱离 tokenizer 直接横向比较。

### 滑动窗口样本并不独立

相邻的 64 字符窗口重叠 63 个字符。55936 个窗口不等于 55936 条独立文本。如果随机拆分窗口做验证，训练和验证会严重泄漏；应先按连续文本区间切分，再分别构造窗口。

### 生成比训练更难并行

训练已知完整目标，可以一次处理所有位置；生成必须等待上一个 token。当前实现又没有 KV cache，每一步都重新计算整个已有上下文，因此生成越长，重复计算越多。

## Sampling 参数

- temperature 小于 1：分布更尖锐，输出更保守。
- temperature 大于 1：分布更平坦，输出更多样但更容易出错。
- top-k：只在概率最高的 k 个 token 中采样。
- 贪心生成：应显式 `argmax`，不能把 temperature 设置为 0；除以 0 会产生非法数值。

## 教学版边界

- 语料只有三个重复短句，低 loss 更多说明记忆了周期模式，不代表自然语言泛化。
- token Embedding 与输出头没有权重共享。
- AdamW 默认对 LayerNorm 和 bias 也做 weight decay。
- checkpoint 没保存架构配置、optimizer 和训练进度。
- 词表没有 `<unk>`；prompt 出现训练词表之外的字符会失败。
- 没有验证集、学习率 warmup/decay、梯度裁剪和 KV cache。
- 每层保存一份 512x512 的浮点因果 mask，会增大 checkpoint；它不是可训练参数，但默认 buffer 会进入 `state_dict`。

### 上下文长度必须一致

可学习位置 Embedding 的大小决定模型真正支持的最大上下文。attention mask 比它更大并不能扩展上下文。生成逻辑必须按模型的 `max_seq_len` 截断，否则位置索引会越界。

## 建议实验

1. 过拟合单个 batch，验证标签错位、mask 和 loss 是否正确。
2. 测试长度 1、63、64、65 的输入边界。
3. 按连续区间建立 train/validation，并绘制两者 perplexity。
4. 换用 Tiny Shakespeare，比较重复 toy 语料与真实文本。
5. 比较字符、byte 和 BPE tokenization。
6. 共享 token Embedding 与 lm_head 权重，比较参数量和验证 loss。
7. 使用 `scaled_dot_product_attention(is_causal=True)`，避免持久化大 mask。
8. 实现 KV cache，比较生成 16、64、256 个 token 的耗时。
9. 比较绝对位置 Embedding、正弦位置编码和 RoPE。

## 产物含义

`gpt_tiny.pt` 保存模型状态，`char_vocab.json` 保存字符与 ID 的映射。二者缺一不可，但仍不足以完全恢复训练，因为架构超参数和 optimizer 状态没有包含在内。
