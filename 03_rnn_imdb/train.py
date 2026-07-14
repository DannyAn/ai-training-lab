"""训练 LSTM 情感分析模型"""

import sys
sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from common.utils import set_seed, get_device
from model import LSTMClassifier


def build_vocab(dataset, max_vocab: int = 25000):
    from collections import Counter
    counter = Counter()
    for example in dataset:
        counter.update(example["text"].split())
    # 保留最常见的词
    most_common = counter.most_common(max_vocab - 2)  # 留出 <pad> 和 <unk>
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def encode(text: str, vocab: dict, max_len: int = 200):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 加载 IMDb 数据集
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb", trust_remote_code=True)
    vocab = build_vocab(dataset["train"])
    print(f"Vocab size: {len(vocab)}")

    def collate_fn(batch):
        texts = [encode(ex["text"], vocab) for ex in batch]
        labels = [ex["label"] for ex in batch]
        return torch.tensor(texts), torch.tensor(labels)

    train_loader = DataLoader(
        dataset["train"], batch_size=64, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset["test"].select(range(2000)), batch_size=64, collate_fn=collate_fn
    )

    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 4):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        # validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}  val_acc={correct/total:.2%}")


if __name__ == "__main__":
    main()
