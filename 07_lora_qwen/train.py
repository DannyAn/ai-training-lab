"""LoRA 微调 Qwen 指令模型

学习目标：从全量微调跨入参数高效微调(PEFT)--只训极少的 adapter 参数就改变大模型行为。
注意：完整运行需 GPU(24GB+，QLoRA 可降至 12GB+)。本机 MPS/CPU 无法跑 4bit 量化(bitsandbytes 仅 Linux CUDA)。
      本文件以"代码结构 + 注释"学习为主，本机跑不起来属正常。
注释标记同 01：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset

from model import create_lora_model

ADAPTER_PATH = "./07_lora_qwen/output/lora"


def prepare_dummy_dataset(tokenizer, num_samples=100):
    """创建一个极小的指令数据集用于演示"""
    data = {
        "instruction": [
            "解释什么是机器学习",
            "用 Python 写一个 hello world",
            "什么是反向传播？",
        ] * (num_samples // 3 + 1),
        "output": [
            "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进，而无需进行明确的编程。",
            '```python\nprint("Hello, World!")\n```',
            "反向传播是一种用于训练神经网络的算法，它通过计算损失函数相对于网络权重的梯度来更新权重。",
        ] * (num_samples // 3 + 1),
    }

    def format_fn(examples):
        # 【记忆】指令微调的数据格式：把"指令 + 回答"拼成模型的对话模板(Qwen 用 <|im_start|>...)。
        #   模型学会"看到这种指令，就生成对应回答"。
        texts = [
            f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n{outp}<|im_end|>"
            for instr, outp in zip(examples["instruction"], examples["output"])
        ]
        tokenized = tokenizer(texts, truncation=True, padding=False, max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # 【记忆】labels=input_ids：语言模型目标就是预测自己
        return tokenized

    dataset = Dataset.from_dict({k: v[:num_samples] for k, v in data.items()})
    dataset = dataset.map(format_fn, remove_columns=data.keys())
    return dataset


def main():
    # 【记忆】LoRA 核心思想：冻结原始大模型全部参数，只在注意力层旁路注入一对小矩阵 A·B(秩 r 很小)。
    #   只训 A、B -> 可训练参数占比常 <1%，效果接近全量微调。这是大模型微调的行业标准。
    # 【调参】use_4bit：QLoRA 把基座 4bit 量化加载以省显存；纯 LoRA 不量化(use_4bit=False)。
    #   lora_r：秩，越大表达越强但参数越多(常用 8/16)；lora_alpha：缩放系数(常设 2*r)。
    model, tokenizer = create_lora_model(use_4bit=False, lora_r=8, lora_alpha=32)

    dataset = prepare_dummy_dataset(tokenizer, num_samples=100)
    split = dataset.train_test_split(test_size=0.1)

    # 【调参】关键训练参数：
    #   per_device_train_batch_size：每卡 batch；显存不够就调小 + 调大 gradient_accumulation_steps 补偿。
    #   gradient_accumulation_steps：累积多步再更新一次，等效"更大 batch"但不爆显存。
    #   bf16：混合精度，省显存加速(需 Ampere+ GPU)；CPU/MPS 不支持，会自动关掉。
    args = TrainingArguments(
        output_dir="./07_lora_qwen/output",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        learning_rate=2e-4,  # 【调参】LoRA 用比全量微调更大的 lr(2e-4)，因为只更新小矩阵
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    # 【记忆】HuggingFace Trainer 封装了训练循环(就是 01 里手写的那套)，省去手写代码。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )
    trainer.train()

    # 【产物】LoRA 只存 adapter(A、B 小矩阵)，通常几百 KB ~ 几 MB，不是全量权重！
    #   基座模型仍在 HF 缓存里不动 -> 分发微调成果只传 adapter 即可，极轻量。
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    print(f"LoRA adapter 已保存到: {ADAPTER_PATH}")

    # 【使用】推理演示。部署标准做法是：
    #     base = AutoModelForCausalLM.from_pretrained(基座名)
    #     infer = PeftModel.from_pretrained(base, ADAPTER_PATH)   # 只靠 adapter 复现微调效果
    # 这里直接用训练好的 model 生成(基座已在内存)，省去重复加载：
    model.eval()
    prompt = "<|im_start|>user\n什么是深度学习？<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100)
    print("生成:", tokenizer.decode(out[0], skip_special_tokens=False))

    # 【练习】
    #   1. 看 create_lora_model 里 print_trainable_parameters() 输出：可训练参数占比通常 <1%。
    #   2. lora_r 从 8 调到 64，看参数量增长(效果略好但更易过拟合、更慢)。
    #   3. 换更大基座(如 Qwen2.5-7B) + use_4bit=True(QLoRA)，在有限显存上微调更大模型。


if __name__ == "__main__":
    main()
