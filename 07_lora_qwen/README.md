# 07 Qwen LoRA：参数高效指令微调

本模块使用 PEFT/LoRA 微调 `Qwen/Qwen2.5-1.5B-Instruct`。它会改变模型行为，但只训练注入的 adapter，不更新完整基座权重。

## 运行与资源

```bash
python 07_lora_qwen/model.py
python 07_lora_qwen/train.py
```

这两个命令都会实际下载并加载 Qwen，不是轻量 shape test。当前 `train.py` 使用 `use_4bit=False`；bitsandbytes 4-bit QLoRA 主要面向 Linux CUDA，macOS/MPS 不适用。

产物保存在 `07_lora_qwen/output/lora/`。

## LoRA 做了什么

对原线性层权重 `W`，LoRA 冻结 `W`，训练两个低秩矩阵：

```text
输出 = W x + (alpha / r) * B A x
```

- `r` 决定低秩通道和 adapter 容量。
- `alpha / r` 决定 LoRA 更新的缩放。
- adapter 可以挂载到同一个基座上，也可以合并后部署。

当前将 LoRA 注入 Qwen 每层的 `q/k/v/o` attention projection 和 `gate/up/down` MLP projection。覆盖范围较广，因此 adapter 不只是“几百 KB”，按当前配置可能达到数十 MB。

## 数据流

```text
instruction + output
-> Qwen chat 格式文本
-> tokenizer
-> input_ids
-> labels = input_ids
-> causal next-token loss
-> 只更新 LoRA adapter
```

训练数据只有三条唯一问答，重复到 100 条；它用于展示 API 与训练流程，不能证明模型获得可靠的新能力。

## 重要知识

### 当前是全序列损失

`labels=input_ids` 表示模板、用户问题和 assistant 回答都参与 loss。生产 SFT 常把 system/user 部分设为 `-100`，只监督 assistant response，避免模型花容量学习复述 prompt。

### LoRA adapter 不能单独推理

adapter 只保存增量。恢复需要：

- 精确基座模型及 revision。
- tokenizer 和 chat template。
- LoRA adapter 与配置。
- 推理生成参数。

除非执行 merge 并保存完整模型，否则 adapter 目录不是独立大模型。

### LoRA 与 QLoRA 不同

- LoRA：基座通常以 fp16/bf16/fp32 加载，基座冻结，训练 adapter。
- QLoRA：基座以 4-bit 量化加载，adapter 仍以较高精度训练，进一步降低权重显存。

梯度累积只增加有效 batch，不会减少单次 forward 的 activation；gradient checkpointing 才主要通过重算降低 activation 显存。

### Chat template 是模型契约

当前代码手工拼接 Qwen special tokens。更稳妥的方法是使用 tokenizer 的 `apply_chat_template()`，避免模型版本变化后训练格式和推理格式不一致。

## 教学版边界

- train/eval 都含有重复的相同问答，验证 loss 无法代表泛化。
- 默认总 update 数约 36，小于 `eval_steps=50`，训练期间可能一次定期 evaluation 都不会发生。
- QLoRA 分支没有完整展示 `prepare_model_for_kbit_training()` 等常见准备步骤。
- `torch.cuda.is_available()` 不等于 GPU 支持 bf16，应检查具体能力。
- 没有固定模型 revision、数据 seed、warmup、最佳 checkpoint 和断点续训。
- `save_strategy="no"` 意味着中断后无法从 Trainer checkpoint 恢复。
- `trust_remote_code=True` 会执行模型仓库代码，生产中应固定 revision 并审查供应链。

## 建议实验

1. 实现 assistant-only loss mask，对比是否减少 prompt 复述。
2. 比较只注入 q/v、attention 全部 projection、attention + MLP。
3. 联合比较 `r` 与 `alpha/r`，不要只改变 r。
4. 改为 epoch evaluation，确保评估真正发生。
5. 按唯一 instruction 分组切分，消除重复数据泄漏。
6. 在 CUDA 环境比较 LoRA 和完整 QLoRA 的峰值显存。
7. 比较运行时挂载 adapter 与 merge 后模型的大小和延迟。
8. 建立训练前后 held-out prompt 评测，检查能力提升与灾难性遗忘。

## 与 RAG 的区别

LoRA 修改模型参数，适合学习输出格式、语气、任务模式或稳定领域行为。RAG 不修改模型参数，而是在回答时提供可更新的外部知识。需要频繁更新事实知识时，通常优先考虑 RAG；需要改变行为模式时，再考虑微调。
