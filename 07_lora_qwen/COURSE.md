# 07 LoRA Qwen 讲师脚本

建议时长：80 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能解释 LoRA 的低秩增量、LoRA 与 QLoRA 的差异、adapter 的恢复依赖，并判断一个微调评估是否发生数据泄漏。

## 开场问题

提问：“如果每个业务任务都复制一份 1.5B 或 7B 模型，存储、训练和部署成本会怎样？”

引出冻结共享基座、为每个任务保存小型 adapter。

## 板书

```text
W' = W + scale * B A
W 冻结
A, B 可训练
rank(A,B) = r << hidden size
scale = alpha / r
```

再画：`base model + adapter A`、`base model + adapter B`。

## 代码走读

1. `model.py`：基座加载与 `use_4bit`。
2. LoRA config：r、alpha、dropout 和七类 target modules。
3. `train.py`：三条唯一数据重复到 100 条。
4. 手写 chat template 与 `labels=input_ids`。
5. 解释当前全序列 loss 与 assistant-only loss。
6. TrainingArguments：micro batch、gradient accumulation、eval_steps。
7. 估算总 update 小于 50，指出默认 evaluation 可能根本不触发。
8. adapter 保存与基座依赖。

## 现场演示策略

无兼容 CUDA 时不现场完整训练。展示：

- 代码路径和预先保存的 `print_trainable_parameters()` 输出。
- base 权重与 adapter 文件大小对比。
- 一张 LoRA/QLoRA 显存构成图。

有 CUDA 且模型已缓存时，可运行短训练副本；不要在课堂临时下载大模型。

## 必讲误区

- LoRA 不是“只微调最后一层”，而是在指定线性层旁路注入矩阵。
- adapter 不能脱离精确基座和 tokenizer 独立推理。
- r 增大时如果 alpha 不变，缩放也会改变。
- 三条数据重复后随机切分会让 train/eval 包含相同问答。
- QLoRA 不只是把 `use_4bit=True`；完整 recipe 还涉及 k-bit training preparation 和硬件支持。
- gradient accumulation 不等同于 gradient checkpointing。

## 课堂实验

给一段 user/assistant token 序列，让学员手工标出哪些 labels 应设为 `-100`，设计 assistant-only loss。再要求按唯一 instruction 划分 train/eval，解释为什么不能先复制后随机切分。

## 提问

1. LoRA adapter 中有没有完整基座参数？
2. LoRA 和 QLoRA 分别节省什么？
3. 为什么当前 eval loss 不能证明泛化？
4. target module 范围和 r 哪个也会显著影响 adapter 大小？

## 收束

微调改变模型行为，但不适合频繁写入新事实。下一节使用外部可更新知识库，让模型在回答时检索资料。
