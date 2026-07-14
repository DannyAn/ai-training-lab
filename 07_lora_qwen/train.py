"""LoRA 微调训练脚本

注意：完整运行需要 GPU (24GB+ 显存 / 使用 QLoRA 可降至 12GB+)
"""

import sys
sys.path.append(".")

import torch
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import Dataset

from model import create_lora_model


def prepare_dummy_dataset(tokenizer, num_samples: int = 100):
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
        texts = [
            f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n{outp}<|im_end|>"
            for instr, outp in zip(examples["instruction"], examples["output"])
        ]
        tokenized = tokenizer(
            texts, truncation=True, padding=False, max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_dict({k: v[:num_samples] for k, v in data.items()})
    dataset = dataset.map(format_fn, remove_columns=data.keys())
    return dataset


def main():
    model, tokenizer = create_lora_model(use_4bit=False)

    dataset = prepare_dummy_dataset(tokenizer, num_samples=100)
    split = dataset.train_test_split(test_size=0.1)

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
        learning_rate=2e-4,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()

    # 保存 LoRA 权重
    model.save_pretrained("./07_lora_qwen/output/lora")
    tokenizer.save_pretrained("./07_lora_qwen/output/lora")
    print("LoRA weights saved to ./07_lora_qwen/output/lora")

    # 测试生成
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    prompt = "<|im_start|>user\n什么是深度学习？<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("\nGenerated:", tokenizer.decode(outputs[0], skip_special_tokens=False))


if __name__ == "__main__":
    main()
