"""训练 CIFAR-10 CNN 分类器

学习目标：在 MNIST 基础上掌握数据增强、BatchNorm、Dropout 三大正则化手段。
注释标记同 01_mnist：【记忆】【调参】【产物】【使用】【练习】
"""

import sys
sys.path.append(".")

import torch
from common.utils import set_seed, get_device
from common.trainer import Trainer
from common.data import get_cifar10_loaders
from model import CIFAR10CNN

MODEL_PATH = "./02_cifar10/cifar10_cnn.pt"


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # 【记忆】数据增强(在 common/data.py 里)：随机裁剪/翻转 -> 让模型见过更多变体，防过拟合。
    #   这是 CV 提分标配：不加增强，CIFAR-10 会迅速过拟合(训练集 95%+ 但测试集只有 70% 左右)。
    # 【调参】batch_size：CIFAR 图比 MNIST 大 3 通道，显存吃紧可调小到 32。
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

    model = CIFAR10CNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # 【记忆】model.py 里三个关键正则化组件，记清各自作用：
    #   BatchNorm：归一化每层激活，稳定训练、允许用更大 lr，是深网能训起来的关键。
    #   Dropout：训练时随机丢弃一部分神经元，强制不依赖单一特征，防过拟合。
    #   AdaptiveAvgPool2d(1)：把任意空间尺寸压成 1x1，让分类头与输入尺寸解耦。

    # 【调参】AdamW 比 Adam 多了权重衰减(weight_decay)，是 CV 微调常用选择。
    #   weight_decay 调大(1e-2)更强抑制过拟合；设 0 则等同普通 Adam。默认写 1e-4，可改成 1e-3 尝试。
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    # 【调参】num_epochs：CIFAR-10 比 MNIST 难，5 epoch 远不够，通常 20~50 起步。
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=20, # epoch 训练时间约 5~10 分钟，视显卡而定,5 epoch 远不够，通常 20~50 起步,默认写20，可改成50尝试。
    )
    history = trainer.fit()
    print(f"\nBest val acc: {max(history['val_acc']):.2%}")

    # 【记忆】测试集评估：训练全程没碰过 test_loader，它是最客观的最终指标(val 也参与了模型选择，略有偏向)。
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Test accuracy: {correct / total:.2%}")

    # 【产物】保存权重。和 01 一样是 state_dict，但模型更深，文件更大。
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型权重已保存到: {MODEL_PATH}")

    # 【使用】推理演示：加载权重对测试图预测。
    #   注意：推理要用验证/测试 transform(只 Normalize，不随机增强)。
    infer = CIFAR10CNN().to(device)
    infer.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    infer.eval()
    x, y = next(iter(test_loader))
    with torch.no_grad():
        preds = infer(x.to(device)).argmax(dim=1)
    print(f"推理示例 - 预测: {preds[:8].tolist()}  真实: {y[:8].tolist()}")

    # 【练习】
    #   1. 在 common/data.py 把 train_transform 的 RandomCrop/RandomFlip 注释掉，重训对比 test_acc(明显下降)。
    #   2. num_epochs 改成 50，观察 train_acc 持续升而 test_acc 停滞/下降(过拟合)。
    #   3. model.py 的 Dropout(0.3) 改成 0 和 0.6，看过拟合程度变化。


if __name__ == "__main__":
    main()
