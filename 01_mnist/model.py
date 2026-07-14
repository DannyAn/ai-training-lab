"""MNIST CNN 模型：卷积 + 池化 + 全连接"""

import torch
from torch import nn


class MNISTCNN(nn.Module):
    """简单 CNN 用于 MNIST 分类 (10 类)"""

    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MNISTCNN()
    dummy = torch.randn(2, 1, 28, 28)
    out = model(dummy)
    print(f"Output shape: {out.shape}  (expected [2, 10])")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
