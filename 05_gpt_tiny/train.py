"""训练微型 GPT 语言模型 (字符级)

学习目标：理解自回归语言模型--因果注意力、next-token 预测、采样生成。
注释标记同 01：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common.utils import set_seed, get_device
from model import GPT

MODEL_PATH = "./05_gpt_tiny/gpt_tiny.pt"
VOCAB_PATH = "./05_gpt_tiny/char_vocab.json"


class CharDataset(Dataset):
    """字符级语言建模：把文本看成 token 序列，目标是预测下一个字符。"""

    def __init__(self, text, seq_len=128):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # 【记忆】语言模型样本构造：x = 前 seq_len 个字符，y = 错位一位的下一个字符。
        #   即用第 1..n 个字符预测第 2..n+1 个，每个位置都在预测"下一个 token"。
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 【记忆】这里用拼凑的短文本演示。经典做法是用 Tiny Shakespeare(莎士比亚全集~1MB)。
    text = "Hello world. " * 1000 + "This is a tiny GPT model. " * 1000 + "AI training lab. " * 1000

    dataset = CharDataset(text, seq_len=64)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 【调参】d_model/n_heads/n_layers 决定模型容量，这里刻意调小(192/3/3)让 CPU/MPS 也能跑。
    model = GPT(vocab_size=dataset.vocab_size, d_model=192, n_heads=3, n_layers=3, max_seq_len=64).to(device)
    print(f"Vocab size: {dataset.vocab_size}, Model params: {sum(p.numel() for p in model.parameters()):,}")

    # 【记忆】GPT 用 3e-4 是 Karpathy 的经验值；语言模型对 lr 敏感，太大极易 loss 爆炸。
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 11):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            # 【记忆】loss 把 (batch, seq, vocab) 摊平成 (batch*seq, vocab)，对齐摊平的 y。
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        # 【记忆】perplexity(困惑度)= exp(loss)，表示模型平均在几个 token 间"犹豫"，越低越好；
        #   1.0 表示完全确定。是语言模型的标准指标。
        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, perplexity={torch.exp(torch.tensor(avg_loss)):.2f}")

    # 【产物】字符级模型要存"权重 + 字符表(stoi/itos)"，和 NLP 词表同理。
    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "w") as f:
        json.dump({"stoi": dataset.stoi, "itos": {str(k): v for k, v in dataset.itos.items()}}, f)
    print(f"权重: {MODEL_PATH}\n词表: {VOCAB_PATH}")

    # 【使用】加载权重重新生成，验证产物能独立工作(不依赖训练时的内存状态)。
    infer = GPT(vocab_size=dataset.vocab_size, d_model=192, n_heads=3, n_layers=3, max_seq_len=64).to(device)
    infer.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    infer.eval()
    prompt = torch.tensor([[dataset.stoi["H"]]], device=device)
    out = infer.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=10)
    print("生成样本:", "".join(dataset.itos[i.item()] for i in out[0]))

    # 【调参·生成】生成时(见 model.py generate)两个关键旋钮：
    #   temperature：>1 更随机有创意，<1 更保守确定，0 退化为贪心。
    #   top_k：每步只在概率最高的 k 个里采样，防止采到离谱 token，常用 10~50。

    # 【练习】
    #   1. 换成 Tiny Shakespeare 文本，多训些 epoch，看生成从乱码变成像样的"莎士比亚腔"。
    #   2. temperature 分别用 0.1 / 1.0 / 2.0 生成，对比随机性与连贯性。
    #   3. 去掉 top_k(传 None)，看是否会偶尔蹦出无意义片段。


if __name__ == "__main__":
    main()
