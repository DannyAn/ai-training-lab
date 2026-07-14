"""训练 CIFAR-10 CNN 分类器"""

import sys
sys.path.append(".")

from common.utils import set_seed, get_device
from common.trainer import Trainer
from common.data import get_cifar10_loaders
from model import CIFAR10CNN


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=64)

    model = CIFAR10CNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=20,
    )
    history = trainer.fit()

    # final test evaluation
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"\nTest accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    main()
