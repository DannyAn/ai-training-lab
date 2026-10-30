# 02 CIFAR-10 讲师脚本

建议时长：75 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能解释数据增强、BatchNorm 和全局平均池化，并发现 Dataset/Subset 共享 transform 带来的验证问题。

## 开场问题

提问：“MNIST 已经能到很高准确率，为什么换成同样只有 10 类的 CIFAR-10 会难很多？”

引导回答：颜色、背景、视角、类内差异、较低分辨率中的真实世界变化。

## 板书

```text
训练图：RandomCrop + Flip + Normalize
验证图：应该固定预处理
测试图：固定预处理，只做最终评价

32x32 -> 16x16 -> 8x8 -> 4x4 -> GAP -> 1x1
通道：3 -> 64 -> 128 -> 256
```

## 代码走读

1. `model.py`：逐 block 解释空间缩小、通道增加。
2. 指出 `AdaptiveAvgPool2d(1)` 如何避免大 Flatten 分类头。
3. `common/data.py`：比较 train/test transform。
4. 重点展示先创建一个带 train transform 的 `full`，再 `random_split`。
5. 让学员判断 val subset 使用哪个 transform，并解释 Subset 只保存索引。
6. `train.py`：AdamW、weight decay、test 和 checkpoint。

## 现场演示

```bash
python 02_cifar10/model.py
```

完整 20 epoch 不建议课堂等待。提前准备完整日志，现场可展示短跑、已下载数据和同一模型重复 validation 的波动。

## 必讲误区

- 数据增强不是简单“数据越多越好”，必须保持标签语义。
- `model.eval()` 不关闭 autograd；测试还应使用 inference mode/no_grad。
- BatchNorm 在 train/eval 使用不同统计量，且小 batch 可能不稳定。
- 当前验证集错误地复用了随机训练增强，这是代码行为，不是推荐实践。
- test 每次被拿来调参后，就不再是严格意义上的最终 test。

## 课堂实验

设计两个 CIFAR Dataset，分别使用 train 和 eval transform，再用同一组固定互斥索引构造 Subset。对同一 checkpoint 连续评估五次，比较修正前后的准确率标准差。

## 提问

1. 为什么 BN 前的 Conv 通常可以 `bias=False`？
2. Dropout 和 weight decay 分别作用在哪里？
3. 为什么打印最佳 val acc 后仍可能保存了较差的模型？

## 收束

图像是固定二维网格，CNN 利用局部结构。文本长度可变且顺序重要，下一节转向序列表示和 padding。
