"""通用工具函数"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置全局随机种子，保证训练结果更可复现。

    这段函数最重要的知识点是：
    1. 随机性不是只靠 PyTorch 一层控制，Python / NumPy / PyTorch 都要一起锁定。
    2. `torch.manual_seed` 是最基础、最稳定的入口，它能让大部分 tensor 初始化行为可复现。
    3. CUDA 还需要单独设置 CUDA 的随机种子，以及 cuDNN 的确定性模式；
       这一步只在真的有 CUDA 时才有意义。
    4. `seed` 本身是“随机数发生器的起点”；`42` 只是常见的固定值，
       并不是唯一正确答案。换成 `52` 或 `2052` 也可以，只要实验过程中保持一致即可。
    """
    # 一、先锁定 Python 和 NumPy 的随机源。
    # 这些是所有实验都要做的“基础复现设置”。
    random.seed(seed)
    np.random.seed(seed)

    # 二、再锁定 PyTorch 的随机种子。
    # 这一步会影响大多数 tensor 初始化、dropout、shuffle 等行为。
    torch.manual_seed(seed)

    # 三、如果当前设备是 CUDA，补充这两项更强的确定性控制。
    # cuDNN 的 deterministic 开关会让卷积/图像相关算子更稳定地复现。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 四、MPS / CPU 不做“伪 CUDA 配置”。
    # 这里保持保守做法：只保留通用的 `torch.manual_seed`，避免调用不存在的 backend API。


def get_device() -> torch.device:
    """自动选择 CUDA / MPS / CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """将秒数格式化为 mm:ss 或 hh:mm:ss"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
