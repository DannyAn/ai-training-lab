"""通用数据加载工具"""

import os

from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

# 锚定到仓库根目录下的 data/，避免因运行目录不同而把数据散落到不同位置、造成重复下载
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_REPO_ROOT, "data")

_MNIST_RAW = os.path.join(DATA_ROOT, "MNIST", "raw")
_MNIST_FILES = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
]

_CIFAR_DIR = os.path.join(DATA_ROOT, "cifar-10-batches-py")
_CIFAR_FILES = [
    "data_batch_1", "data_batch_2", "data_batch_3",
    "data_batch_4", "data_batch_5", "test_batch",
]


def _files_present(folder: str, files: list[str]) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in files)


def _log_data_status(name: str, folder: str, files: list[str]) -> None:
    if _files_present(folder, files):
        print(f"{name}: 已找到本地数据 {folder}，跳过下载")
    else:
        print(f"{name}: 本地数据缺失，将下载到 {folder}")


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

    _log_data_status("MNIST", _MNIST_RAW, _MNIST_FILES)
    full = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
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

    _log_data_status("CIFAR-10", _CIFAR_DIR, _CIFAR_FILES)
    full = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    val_size = int(len(full) * val_split)
    train_set, val_set = random_split(full, [len(full) - val_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_set = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=val_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
