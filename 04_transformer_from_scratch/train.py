"""训练 Transformer - 德语->英语翻译 (玩具数据集)

学习目标：亲手实现并跑通 Transformer 训练循环，理解注意力、位置编码、因果掩码、teacher forcing。
注意：完整训练需真实平行语料 + 多 GPU 长时间训练。这里用随机数据演示"循环结构"，
      模型学不到真实翻译--重点是看清数据怎么流过模型，而不是结果。
注释标记同 01：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from common.utils import set_seed, get_device
from model import Transformer, generate_square_subsequent_mask

MODEL_PATH = "./04_transformer_from_scratch/transformer_demo.pt"


class TinyTranslationDataset(Dataset):
    """极小英德翻译数据集，仅演示训练循环(数据随机，无真实语义)"""

    def __init__(self, size=1000, max_len=20):
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
    # 【调参】Transformer 核心超参(在 model.py 构造里)：
    #   d_model：隐层维度(越大容量越大、越慢)；n_heads：注意力头数(必须整除 d_model)；
    #   n_layers：编码器/解码器各堆几层；d_ff：前馈层中间维度(通常 4*d_model)。
    model = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        d_model=128, n_heads=4, n_layers=3, d_ff=512,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset = TinyTranslationDataset(size=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 【记忆】seq2seq 的 loss：ignore_index=0 让 padding 位置不参与 loss，否则模型会花力气学补齐符。
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(1, 4):
        total_loss = 0
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            # 【记忆】teacher forcing：训练时把"真值右移一位"作为解码器输入(tgt_in)，
            #   让它预测下一位(tgt_out)，而不是用自己上一步的预测--训练快且稳。
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            # 【记忆】因果掩码(causal mask)：解码位置 i 只能看到 <=i，看不到未来，防止"偷看答案"。
            mask = generate_square_subsequent_mask(tgt_in.size(1), device)
            logits = model(src, tgt_in, tgt_mask=mask)

            loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
            optim.zero_grad()
            loss.backward()
            # 【记忆】梯度裁剪：Transformer 对梯度爆炸极敏感，训练初期很容易 loss 变 NaN。
            #   clip_grad_norm_ 把总梯度范数压到 max_norm 以下，是 Transformer 训练的标配。
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    # 【产物】存权重。真正推理还需存 src/tgt 词表(BPE/wordpiece)，toy 数据这里省略。
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"权重已保存到: {MODEL_PATH}")

    # 【使用】贪心解码演示：自回归逐 token 生成。
    #   推理与训练最大区别：没有真值可喂，解码器每步吃"自己上一步输出"，循环生成。
    model.eval()
    src = torch.randint(2, 50, (1, 8)).to(device)  # 假装一句源语言
    tgt = torch.tensor([[1]], device=device)       # 起始符 <bos>=1
    with torch.no_grad():
        for _ in range(12):
            mask = generate_square_subsequent_mask(tgt.size(1), device)
            logits = model(src, tgt, tgt_mask=mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # 取最后一位概率最大的 token
            tgt = torch.cat([tgt, next_token], dim=1)
    print(f"贪心解码 token 序列: {tgt[0].tolist()}  (toy 模型，无真实语义)")

    # 【练习】
    #   1. 换成真实平行语料(如 WMT 英德)，能跑出真正翻译(需大量算力)。
    #   2. n_heads 从 4 改成 1，对比参数量与效果(单头 vs 多头)。
    #   3. 去掉因果掩码，观察 loss 异常下降--模型偷看了未来答案。


if __name__ == "__main__":
    main()
