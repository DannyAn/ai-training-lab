"""训练 LSTM 情感分析模型 (IMDb)

学习目标：从图像跨入 NLP，掌握 Embedding、LSTM、词表、序列 padding。
注释标记同 01：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from common.utils import set_seed, get_device
from model import LSTMClassifier

MODEL_PATH = "./03_rnn_imdb/imdb_lstm.pt"
VOCAB_PATH = "./03_rnn_imdb/imdb_vocab.json"


def build_vocab(dataset, max_vocab=25000):
    from collections import Counter
    counter = Counter()
    for example in dataset:
        counter.update(example["text"].split())
    most_common = counter.most_common(max_vocab - 2)  # 留出 <pad> 和 <unk>
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(most_common)}
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1
    return vocab


def encode(text, vocab, max_len=200):
    # 【记忆】NLP 预处理三步：分词 -> 查词表转 id -> 截断/补齐到固定长度。
    #   padding 是为了让一个 batch 里长短不一的句子能拼成张量。
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 【记忆】IMDb：5 万条电影评论，二分类(正面/负面)，NLP 入门经典数据集。
    #   load_dataset 首次会从 HuggingFace 下载到本地缓存(需要网络)。
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb", trust_remote_code=True)
    vocab = build_vocab(dataset["train"])
    print(f"Vocab size: {len(vocab)}")

    def collate_fn(batch):
        texts = [encode(ex["text"], vocab) for ex in batch]
        labels = [ex["label"] for ex in batch]
        return torch.tensor(texts), torch.tensor(labels)

    train_loader = DataLoader(dataset["train"], batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset["test"].select(range(2000)), batch_size=64, collate_fn=collate_fn)

    # 【记忆】模型结构(见 model.py)：Embedding(词id->词向量) -> 双向 LSTM -> 取最后层隐状态 -> 分类头。
    #   bidirectional=True：正向+反向各读一遍，情感判别双向信息都有用。
    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 【调参】lr：LSTM 用 1e-3 起步；若 loss 不降可试 5e-4。RNN 类模型对 lr 敏感，大了容易梯度爆炸。
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 【调参】num_epochs：文本分类 3~5 epoch 通常够，过多易过拟合。
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

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"Epoch {epoch}: train_loss={total_loss/len(train_loader):.4f}  val_acc={correct/total:.2%}")

    # 【产物】NLP 模型必须同时存"权重 + 词表"，缺一不可！
    #   只有权重没词表 -> 新文本无法转成 id -> 没法推理。这是 NLP 部署最常见的坑。
    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump(vocab, f)
    print(f"权重: {MODEL_PATH}\n词表: {VOCAB_PATH}")

    # 【使用】推理演示：输入一句影评，预测情感。
    infer = LSTMClassifier(vocab_size=len(vocab)).to(device)
    infer.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    infer.eval()

    def predict(text):
        ids = torch.tensor([encode(text, vocab)]).to(device)
        with torch.no_grad():
            pred = infer(ids).argmax(dim=1).item()
        return "正面" if pred == 1 else "负面"

    for s in ["This movie is absolutely wonderful and touching",
              "Boring, slow and a complete waste of time"]:
        print(f"  [{predict(s)}] {s}")

    # 【练习】
    #   1. model.py 的 bidirectional 改成 False，对比 val_acc(下降，单向漏掉后文信息)。
    #   2. embed_dim 从 128 调到 32 和 512，看欠/过拟合。
    #   3. max_len 从 200 调到 20，看长评论被截断后准确率下降。


if __name__ == "__main__":
    main()
