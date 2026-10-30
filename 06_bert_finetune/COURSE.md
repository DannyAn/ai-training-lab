# 06 BERT 微调讲师脚本

建议时长：80 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能解释预训练与微调、BERT 双向 attention、attention mask、全量微调与冻结，并知道模型与 tokenizer 必须一起保存。

## 开场问题

提问：“如果情感训练集只有几万句，为什么可以训练 1.1 亿参数模型，而不必从随机参数开始？”

引出迁移学习：预训练提供通用语言表示，下游训练只做适配。

## 板书

```text
sentence
-> WordPiece
-> [CLS] tokens [SEP] [PAD]
-> input_ids + attention_mask
-> bidirectional BERT encoder
-> pooled representation
-> classifier -> 2 logits
```

对比 GPT：BERT 非 pad token 双向互看；GPT 只能看左侧。

## 代码走读

1. `model.py`：`BertModel.from_pretrained` 和分类头。
2. 解释 `pooler_output` 是 CLS 经 Linear+tanh，不是原始 CLS。
3. `train.py`：Dataset map、固定 128 padding、attention mask。
4. 指出单句任务未传 token_type_ids 通常无害。
5. 手写 AdamW 循环：本模块使用 HF 模型，但没有使用 HF Trainer。
6. 保存权重与 tokenizer，讨论离线恢复仍缺什么。

## 现场演示

```bash
python 06_bert_finetune/model.py
```

BERT 全量训练耗时较长，使用课前缓存和预先日志。现场重点展示 tokenizer 对一句话输出的 tokens、input IDs、attention mask，而不是等待三轮训练。

## 必讲误区

- 预训练模型不是不可修改的特征提取器；当前全部参数都更新。
- 冻结后不能机械沿用全量微调的同一个学习率做公平比较。
- 固定 padding 到 128 可能浪费大量计算。
- `state_dict` 不含清晰模型配置、label mapping 和预处理契约。
- validation 只有 872 条，单 seed 小差异未必有意义。
- BERT 适合编码分类，不等于不能参与生成系统，但它本身不是当前 GPT 式生成器。

## 课堂实验

冻结 BERT 只训练分类头，并将分类头学习率调高。比较可训练参数量、step 时间、显存和 validation accuracy。要求说明该实验改变了两个相关因素：训练参数范围和合适学习率。

## 提问

1. attention mask 与 GPT causal mask 有何不同？
2. 为什么 tokenizer 也属于模型产物？
3. `pooler_output` 与 `last_hidden_state[:,0]` 有何区别？
4. 为什么 BERT 微调常需要 warmup？

## 收束

全量微调需要为每个任务更新和保存完整模型。下一节 LoRA 冻结基座，仅学习小型参数增量。
