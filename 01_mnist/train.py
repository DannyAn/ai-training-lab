"""训练 MNIST CNN 分类器

学习目标：跑通"数据 → 模型 → 训练 → 保存 → 推理"完整闭环。
注释标记说明（后续模块沿用同一套）：
  【记忆】必须刻进脑子里的核心概念 / 训练流程
  【调参】可以改的超参数，以及改大/改小会发生什么
  【产物】训练完会生成什么文件、里面装的是什么
  【使用】训练好的模型怎么加载、怎么拿来预测
"""

import sys
sys.path.append(".")

import torch
from common.utils import set_seed, get_device
from common.trainer import Trainer
from common.data import get_mnist_loaders
from model import MNISTCNN


def main():
    # 【记忆】可复现性：固定随机种子，保证每次跑的初始化/数据顺序一致，方便对比调参效果。
    set_seed(42)

    # 【记忆】设备选择：cuda(NVIDIA) > mps(Apple Silicon) > cpu。训练速度差异巨大，有 GPU 一定用 GPU，微软：DirectX。
    device = get_device()
    print(f"Device selected: {device} [CUDA:(Compute Unified Device Architecture),MPS:(Metal Performance Shaders),CPU]")

    # 【调参】batch_size：每一步喂给模型的样本数, 默认程序赋值：64。
    #   - 调大(128/256)：训练更快、梯度更稳，但吃显存，且可能泛化变差。
    #   - 调小(16/32)：显存友好、梯度带噪(有时反而帮助泛化)，但更慢。
    # 【记忆】DataLoader 自动做三件事：打乱顺序(shuffle) / 分批 / 多进程加载(num_workers)。
    train_loader, val_loader = get_mnist_loaders(batch_size=256)

    # 【记忆】模型就是"一堆带参数的算子"。参数量 ≈ 模型容量。
    #   参数太少 → 欠拟合(学不会)；太多 → 过拟合(死记硬背，验证集反而变差)。
    model = MNISTCNN()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 【调参】学习率 lr（learning ratio）：最关键的超参数。
    #   - 太大：loss 震荡甚至爆炸(NaN)；太小：收敛极慢。
    #   - 经验：Adam 起步用 1e-3；调参时按 3x/10x 量级跳(如 3e-4、1e-4)。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 【调参】num_epochs：把训练集完整过几遍。
    #   - 太少：没学够，欠拟合。
    #   - 太多：验证集准确率先升后降(过拟合)，应在该停时停(早停)。
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
    )

    # 【记忆】训练循环四步（每个 batch 重复一次，trainer.fit() 内部就是它）：
    #   1. forward : out = model(x)        前向算预测
    #   2. loss    : l = criterion(out, y) 算预测和真值的误差
    #   3. backward: l.backward()          反向传播求每个参数的梯度
    #   4. step    : optimizer.step()      用梯度更新参数；下次前要 zero_grad() 清掉旧梯度
    history = trainer.fit()

    print(f"\nDone! Best val acc: {max(history['val_acc']):.2%}")

    # ===== 以下两段原代码没有，是补上的"保存 + 使用"闭环 =====

    # 【产物】保存训练结果。state_dict 是"模型所有参数权重的字典{层名: 权重张量}"。
    #   .pt/.pth 是 PyTorch 惯例，本质是 pickle 文件。
    #   这里只存权重(不含模型结构代码) → 加载时必须先有 MNISTCNN 类定义。
    #   另一种存法 torch.save(model) 会连结构一起存，但不推荐(耦合代码版本，易踩坑)。
    save_path = "./01_mnist/mnist_cnn.pt"
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存到: {save_path}")

    # 【使用】推理演示：重建模型 → 加载权重 → eval() → 喂输入 → 得预测。
    #   部署/预测时跑的就是这一段：拿训练好的权重，对新数据做推断。
    infer_model = MNISTCNN().to(device)
    # 【记忆·安全】torch.load 默认走 pickle，加载不可信 .pt 可执行任意代码(供应链/反序列化漏洞)。
    #   只加载自己训练的权重时加 weights_only=True，限定只反序列化张量，从根上杜绝代码执行。
    infer_model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    infer_model.eval()  # 【记忆】推理必须调 eval()：关掉 Dropout、固定 BatchNorm 统计量

    x, y = next(iter(val_loader))  # 借一批验证数据当示例
    x = x.to(device)
    with torch.no_grad():  # 【记忆】推理不需要梯度，no_grad 省显存、加速
        preds = infer_model(x).argmax(dim=1)  # argmax 取概率最大的类别
    print(f"推理示例 - 预测: {preds[:8].tolist()}  真实: {y[:8].tolist()}")

    # 【练习】建议动手试一遍，建立直觉（改完直接 python 01_mnist/train.py 重跑）：
    #   1. num_epochs 改成 15，观察 val_acc 先升后降 → 亲眼看到过拟合。
    #   2. batch_size 分别改成 8 和 256，对比同 epoch 下的速度与准确率。
    #   3. lr 改成 1e-1 看 loss 变 NaN(爆炸)；改成 1e-5 看收敛极慢。
    #   4. 去掉 model.py 里的 Dropout(0.5)，看过拟合是否加重。


if __name__ == "__main__":
    main()
