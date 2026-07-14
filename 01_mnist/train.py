"""训练 MNIST CNN 分类器"""

import sys
sys.path.append(".")

from common.utils import set_seed, get_device
from common.trainer import Trainer
from common.data import get_mnist_loaders
from model import MNISTCNN


def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader = get_mnist_loaders(batch_size=64)

    model = MNISTCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=5,
    )
    history = trainer.fit()

    print(f"\nDone! Best val acc: {max(history['val_acc']):.2%}")


if __name__ == "__main__":
    main()
