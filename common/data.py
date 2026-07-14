"""通用数据加载工具"""

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


def get_mnist_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
):
    """获取 MNIST 训练/验证 DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    val_size = int(len(full) * val_split)
    train_set, val_set = random_split(full, [len(full) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_cifar10_loaders(
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
):
    """获取 CIFAR-10 训练/验证 DataLoader（含数据增强）"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    val_size = int(len(full) * val_split)
    train_set, val_set = random_split(full, [len(full) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
