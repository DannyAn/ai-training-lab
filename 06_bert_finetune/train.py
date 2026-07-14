"""微调 BERT 进行情感分类"""

import sys
sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from common.utils import set_seed, get_device
from model import BertClassifier, load_tokenizer


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 加载 SST-2 数据集
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2", trust_remote_code=True)
    tokenizer = load_tokenizer()

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"], truncation=True, padding="max_length", max_length=128
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=32)

    model = BertClassifier().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

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

        # evaluate
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


if __name__ == "__main__":
    main()
