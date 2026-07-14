"""LSTM 情感分类模型"""

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Embedding + LSTM + 全连接 用于文本分类"""

    def __init__(
        self,
        vocab_size: int = 25000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embed_dim)
        _, (hn, _) = self.lstm(emb)
        # hn: (num_layers * 2, batch, hidden_dim)
        hn = hn[-2:, :, :]  # take last layer forward + backward
        out = hn.permute(1, 0, 2).contiguous().view(x.size(0), -1)
        return self.classifier(out)


if __name__ == "__main__":
    model = LSTMClassifier()
    dummy = torch.randint(0, 1000, (2, 200))
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
