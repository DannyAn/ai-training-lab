"""微型 GPT — decoder-only 因果语言模型"""

import math
import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # (bias) causal mask
        self.register_buffer("bias", torch.tril(torch.ones(1, 1, 512, 512)).view(1, 1, 512, 512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = att.softmax(dim=-1)
        att = self.dropout(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GPTBlock(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """微型 GPT 语言模型"""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 6,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.inference_mode()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int | None = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -512:]
            logits = self(idx_cond)[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            probs = logits.softmax(dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    model = GPT(vocab_size=256)
    dummy = torch.randint(0, 100, (2, 32))
    out = model(dummy)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
