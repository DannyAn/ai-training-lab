"""
训练 Transformer — 德语→英语翻译 (玩具数据集)

注意：完整训练需要多 GPU ≈ 数十万步。这里仅演示训练循环的结构。
"""

import sys
sys.path.append(".")

import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common.utils import set_seed, get_device
from model import Transformer, generate_square_subsequent_mask


class TinyTranslationDataset(Dataset):
    """极小的英德翻译数据集，仅用于演示训练循环"""

    def __init__(self, size: int = 1000, max_len: int = 20):
        self.data = []
        for i in range(size):
            src_len = torch.randint(5, max_len - 2, (1,)).item()
            tgt_len = torch.randint(5, max_len - 2, (1,)).item()
            src = torch.randint(2, 50, (src_len,))
            tgt = torch.randint(2, 50, (tgt_len,))
            self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_idx=0):
    src_seqs = [item[0] for item in batch]
    tgt_seqs = [item[1] for item in batch]
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    vocab_size = 50
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = TinyTranslationDataset(size=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(1, 4):
        total_loss = 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            mask = generate_square_subsequent_mask(tgt_in.size(1), device)
            logits = model(src, tgt_in, tgt_mask=mask)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")


if __name__ == "__main__":
    main()
