"""微调 BERT 进行情感分类 (GLUE SST-2)

学习目标：从"从零训练"跨入"站在巨人肩膀上"--预训练 + 微调范式。
注释标记同 01：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from common.utils import set_seed, get_device
from model import BertClassifier, load_tokenizer

MODEL_PATH = "./06_bert_finetune/bert_sst2.pt"
TOKENIZER_DIR = "./06_bert_finetune/tokenizer"


def main():
    # 【记忆】固定随机种子（seed=42）
    #   让随机初始化、打乱顺序、Dropout 等行为在每次运行时都从同一个起点开始，
    #   这样实验更容易复现，结果也更稳定。
    #   注意：42 只是一个常见示例值，不是唯一正确值。
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 【记忆】SST-2：GLUE 里的句子情感二分类，BERT 微调的经典 benchmark。
    #   新版 datasets(5.x) 要求"命名空间/名字"格式；旧的 load_dataset("glue",...) 脚本式加载已废弃会报错。
    dataset = load_dataset("nyu-mll/glue", "sst2")
    
    # 【记忆】tokenizer 把文本变成模型能吃的 input_ids + attention_mask。
    #   BERT 用 WordPiece 子词分词；padding="max_length" 统一到 128 方便批量。
    tokenizer = load_tokenizer()

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=32)

    model = BertClassifier().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # 【记忆】BertClassifier = 预训练 BERT(约 1.1 亿参数) + 一个线性分类头。
    #   微调时整个模型一起更新；也可以冻结 BERT 只训头(那是"特征提取"玩法，见练习)。

    # 【调参】lr=2e-5：微调预训练模型必须用很小的 lr！
    #   从零训练用 1e-3，但微调用 1e-3 会把预训练权重冲坏。2e-5~5e-5 是 BERT 微调惯例。
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 【调参】num_epochs：微调通常 2~5 epoch 就够，过多易过拟合(预训练模型已很强)。
    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            ids, mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            optim.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                ids, mask, labels = (
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["label"].to(device),
                )
                preds = model(ids, mask).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f"Epoch {epoch}: loss={total_loss/len(train_loader):.4f}  val_acc={correct/total:.2%}")

    # 【产物】BertClassifier 是自定义 nn.Module，存它的 state_dict(含 BERT 权重 + 分类头)。
    #   同时存 tokenizer--部署时和权重一起带走。
    #   (HF 另一惯用法：用 BertForSequenceClassification + save_pretrained，存成模型目录)
    torch.save(model.state_dict(), MODEL_PATH)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"权重: {MODEL_PATH}\n分词器: {TOKENIZER_DIR}")

    # 【使用】推理：加载权重 + tokenizer，对一句话预测情感。
    #   label: 0=负面, 1=正面
    infer = BertClassifier().to(device)
    infer.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    infer.eval()
    sentences = ["a masterpiece of subtle storytelling", "dull and pointless waste of film"]
    enc = tokenizer(sentences, truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        preds = infer(enc["input_ids"].to(device), enc["attention_mask"].to(device)).argmax(dim=1)
    for s, p in zip(sentences, preds.tolist()):
        print(f"  [{'正面' if p == 1 else '负面'}] {s}")

    # 【练习】
    #   1. 在 model.py BertClassifier 里冻结 BERT(对 self.bert 参数设 requires_grad=False)，只训分类头，对比速度与精度。
    #   2. lr 改成 1e-3 看准确率崩塌(冲坏预训练权重)。
    #   3. 换个分类任务(如 5 星评分，num_classes=5)，体会迁移学习的通用性。


if __name__ == "__main__":
    main()
