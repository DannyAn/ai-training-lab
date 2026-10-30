# 02 CIFAR-10：更深的 CNN 与泛化

本模块从灰度数字升级到 RGB 自然图像，引入数据增强、BatchNorm、Dropout、AdamW 和全局平均池化。

## 运行

```bash
python 02_cifar10/model.py
python 02_cifar10/train.py
```

训练产物是 `02_cifar10/cifar10_cnn.pt`。

## 模型和数据流

输入为 `[batch, 3, 32, 32]`：

```text
3x32x32
-> 两层 64 通道 Conv-BN-ReLU -> MaxPool -> 64x16x16
-> 两层 128 通道 Conv-BN-ReLU -> MaxPool -> 128x8x8
-> 两层 256 通道 Conv-BN-ReLU -> MaxPool -> 256x4x4
-> AdaptiveAvgPool(1) -> 256
-> Dropout -> Linear(256,10)
```

训练图像经过随机裁剪和水平翻转；测试图像只做张量转换与标准化。

## 重要知识

### 数据增强改变的是输入分布

随机裁剪和翻转不会创建新的人工标签，而是在保持语义不变的前提下构造输入变体。它通常降低训练集拟合速度，但改善未见图像上的泛化。

### BatchNorm 有参数也有状态

BatchNorm 的 scale 和 bias 是可训练参数，`running_mean` 和 `running_var` 是随训练更新的 buffer。训练时使用 batch 统计量，`eval()` 时使用累计统计量。因此 `model.eval()` 不只是关闭 Dropout，也会改变 BatchNorm 行为。

### 全局平均池化让分类头更小

`AdaptiveAvgPool2d(1)` 将每个通道压成一个数，使分类头只需 `Linear(256, 10)`。如果直接 Flatten `256x4x4`，分类头参数更多，也更依赖固定输入尺寸。

### AdamW 的 weight decay 不是 Dropout

Dropout 正则化激活；weight decay 约束参数。当前代码把 weight decay 应用于所有参数，工程上常把 bias 和 BatchNorm 参数放入不衰减的参数组。

## 当前实现中最值得注意的问题

训练集和验证集由同一个带训练 transform 的 CIFAR-10 dataset 通过 `random_split()` 得到，因此当前验证集实际上也会随机裁剪和翻转。结果是：

- 每次验证看到的图像可能不同。
- 同一模型重复验证的指标可能波动。
- 注释中“验证只做 Normalize”与实际行为不一致。

严谨做法是让训练和验证使用两个 transform 不同、但索引互斥的 dataset/Subset。这个问题很适合作为理解 Dataset、Subset 和 transform 所属位置的练习。

## 教学版边界

- 打印的是历史最佳验证准确率，测试和保存的却是最后一轮模型。
- 最终测试循环调用了 `eval()`，但没有关闭 autograd；`eval()` 不等于 `no_grad()`。
- 没有 scheduler，虽然更完整的 CIFAR 训练通常会使用 cosine 或 step decay。
- 没有训练 accuracy，无法直接观察 train/validation gap。
- BatchNorm 在很小 batch 下统计量可能不稳定。
- checkpoint 不包含训练配置和 optimizer state。

## 建议实验

1. 首先修正验证 transform，然后对同一 checkpoint 连续验证五次，比较修正前后的波动。
2. 分别去掉 RandomCrop、RandomHorizontalFlip、BatchNorm 和 Dropout，做消融实验。
3. 比较 BatchNorm 与 GroupNorm 在 batch size 8 下的表现。
4. 给 BN 前的卷积设置 `bias=False`，比较参数量和准确率。
5. 加入 CosineAnnealingLR，比较固定学习率与调度后的曲线。
6. 保存 best、last 两个 checkpoint，最后只使用 test set 做一次模型比较。

## 从教学到工程

生产训练还需要固定验证预处理、实验配置记录、断点续训、自动混合精度、数据加载性能设置、指标监控和可重复的最终测试协议。模型更深并不自动等于工程更成熟。
