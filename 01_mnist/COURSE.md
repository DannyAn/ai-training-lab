# 01 MNIST 讲师脚本

建议时长：70 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能够口述一个 batch 的训练四步，区分 logits 与概率，并解释保存权重后为什么仍需要模型类定义。

## 开场问题

展示一张 28x28 手写数字，提问：“计算机最终要输出什么？十个概率、一个类别，还是一套规则？”

先给结论：模型输出十个 logits，loss 用标签评价它，优化器依据梯度修改参数。

## 板书

```text
x [B,1,28,28]
-> CNN
-> logits [B,10]
-> CrossEntropy(logits, y)
-> loss 标量
-> backward 得到每个参数的 gradient
-> optimizer.step 修改参数
```

补充写出：`train != validation != inference`。

## 代码走读

1. 打开 `model.py`：沿两次 Conv/Pool 写出 `28 -> 14 -> 7`。
2. 指出模型末尾没有 Softmax，询问为什么。
3. 打开 `train.py` 的设备、DataLoader、optimizer 和 Trainer 构造。
4. 打开 `common/trainer.py`，让学员寻找 forward、loss、backward、step、zero_grad。
5. 回到保存与推理段，解释 `state_dict`、`eval()` 和 `no_grad()`。

## 现场演示

```bash
python 01_mnist/model.py
python 01_mnist/train.py
```

演示时让学员先预测：输出 shape、参数量数量级、loss 方向、accuracy 方向和产物路径。

若完整训练耗时过长，提前准备最后两轮日志；课堂至少现场跑一个 batch 或一个缩短 epoch 的副本，不临时改主分支代码。

## 必讲误区

- `argmax` 作用的是 logits，注释口语可说“最高分”，不应说模型已经输出概率。
- 固定随机种子提高可复现性，不保证跨硬件完全一致。
- 当前 validation 来自训练集切分，不是官方 test。
- 打印历史 best accuracy 不代表保存了 best checkpoint；当前保存的是 last。
- Dropout 不减少参数，只改变训练时激活。

## 课堂实验

将 Dropout 从 0.5 攓为 0，并只训练少量 epoch。要求学员先写假设，再比较 train loss 和 val accuracy。强调一次只改一个变量。

## 提问

1. 为什么 loss.backward() 不直接修改权重？
2. 为什么推理前要 `eval()`，仅使用 `no_grad()` 是否足够？
3. `.pt` 文件是否包含 `MNISTCNN` Python 类？

## 收束

本节建立的是所有后续训练的共同骨架。下一节不改变分类训练本质，而是提高数据难度和模型正则化能力。
