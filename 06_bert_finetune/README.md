# 06 BERT：预训练模型微调

本模块使用预训练 `google-bert/bert-base-uncased` 完成 SST-2 英文情感二分类，展示“预训练通用表示 + 下游任务适配”。

## 运行

```bash
python 06_bert_finetune/model.py
python 06_bert_finetune/train.py
```

首次运行会下载 BERT、tokenizer 和 GLUE/SST-2。训练产物是 `bert_sst2.pt` 与 `tokenizer/`。

## 数据和模型流程

```text
英文句子
-> BERT WordPiece tokenizer
-> [CLS] tokens [SEP] [PAD...]
-> input_ids + attention_mask
-> 12 层 BERT Encoder
-> pooler_output
-> Dropout + Linear(768,2)
-> 情感 logits
```

当前是全量微调：BERT backbone 和分类头都会更新，不是只训练分类头。

## 重要知识

### 微调不是从头训练 BERT

BERT 已通过大规模预训练学到语言表示。SST-2 使用较小学习率调整全部参数，让表示适应情感分类。随机初始化分类头则从零学习标签边界。

### `pooler_output` 不是原始 CLS 向量

它是最后一层 `[CLS]` hidden state 再经过 Linear 和 tanh 的结果。也可以比较原始 CLS 或按 attention mask 做 mean pooling；pooler 不保证对所有下游任务最优。

### attention mask 不是 causal mask

BERT mask 主要屏蔽 `[PAD]`，非 padding token 可以双向查看整句。GPT causal mask 则屏蔽未来位置。这解释了 BERT 更适合编码理解、GPT 更适合生成。

### 单句任务可以省略 token type IDs

SST-2 每条数据只有一句话，默认全零 segment IDs 通常正确。若换成 MRPC、MNLI 等句对任务，应保留 A/B 句段信息。

### 动态 padding 更高效

当前训练把所有句子补到长度 128，大量计算可能浪费在 PAD 上。`DataCollatorWithPadding` 可以按当前 batch 最长句动态补齐，通常降低显存和计算量。

## 全量微调与冻结

冻结 BERT 只训练分类头时，可训练参数从约 1.1 亿降到 1538，但这不是公平地只改一个布尔值：

- 分类头通常需要比 `2e-5` 更大的学习率。
- 冻结会减少 backward 显存与时间。
- 全量微调通常有更强任务适配能力。
- 还可以只解冻最后若干层，或采用分层学习率衰减。

## 教学版边界

- 使用手写 PyTorch 循环，不是 Hugging Face Trainer。
- 没有 warmup 和学习率衰减，而 BERT 微调通常使用 warmup + linear decay。
- 保存最后一轮，不是验证集最优 checkpoint。
- 只报告 accuracy，没有 validation loss、错误分析和多 seed 方差。
- 当前 PyTorch AdamW 对 bias 和 LayerNorm 也应用默认 weight decay。
- `state_dict` 不是自描述的 Hugging Face 模型目录，缺少模型配置、标签映射和预处理契约。
- 恢复时先执行 `BertModel.from_pretrained()`，即使已有 `.pt`，离线环境仍需要 HF cache 或完整本地基座。
- 保存 tokenizer 后，当前演示没有真正从保存目录重新加载它。

## 建议实验

1. 比较冻结、解冻最后 1/3/6 层和全量微调。
2. 比较固定 padding 与动态 padding 的显存、吞吐和准确率。
3. 加入 10% warmup + linear decay。
4. 比较 `1e-5`、`2e-5`、`3e-5`、`5e-5`，并至少运行三个 seed。
5. 比较 pooler、原始 CLS 和 masked mean pooling。
6. 统计长度超过 64/128 的截断比例，再选择 max length。
7. 输出误分类句子，分析否定、反讽、长句和截断问题。
8. 使用 `save_pretrained()` 保存完整模型目录，并在离线模式验证恢复。

## 从预训练到参数高效微调

本模块更新 BERT 的全部参数。下一模块 LoRA 会冻结大模型基座，只训练少量低秩 adapter，以显著降低训练显存和产物大小。
