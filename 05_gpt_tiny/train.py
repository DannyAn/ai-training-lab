"""训练微型 GPT 语言模型"""

import sys
sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common.utils import set_seed, get_device
from model import GPT


class CharDataset(Dataset):
    """字符级语言建模数据集"""

    def __init__(self, text: str, seq_len: int = 128):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.seq_len = seq_len

        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 使用小文本训练
    text = "Hello world. " * 1000 + "This is a tiny GPT model. " * 1000 + "AI training lab. " * 1000

    dataset = CharDataset(text, seq_len=64)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = GPT(vocab_size=dataset.vocab_size, d_model=192, n_heads=3, n_layers=3, max_seq_len=64).to(device)
    print(f"Vocab size: {dataset.vocab_size}, Model params: {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, 11):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:2d}: loss={avg_loss:.4f}, perplexity={torch.exp(torch.tensor(avg_loss)):.2f}")

        # 每 5 个 epoch 生成一段文本
        if epoch % 5 == 0:
            prompt = torch.tensor([[dataset.stoi["H"]]], device=device)
            generated = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=10)
            text = "".join(dataset.itos[i.item()] for i in generated[0])
            print(f"  Sample: {text[:80]}...")


if __name__ == "__main__":
    main()
