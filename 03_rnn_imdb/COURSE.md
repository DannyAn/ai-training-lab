# 03 RNN IMDb 讲师脚本

建议时长：75 分钟。课前阅读：[README.md](./README.md)。

## 本节目标

学员能画出文本到 token ID、Embedding、LSTM 和分类 logits 的流程，并解释为什么 `padding_idx` 不代表 RNN 忽略 padding。

## 开场问题

写出 `good` 和 `bad` 两个字符串，提问：“神经网络为什么不能直接理解这几个字符？把 good 编成 ID 100，会不会比 bad 的 ID 20 更积极？”

强调 ID 是索引，不是数值特征。

## 板书

```text
text -> tokenize -> vocab lookup -> [token IDs]
-> pad/truncate -> [B,200]
-> Embedding -> [B,200,128]
-> BiLSTM -> hidden states
-> forward/backward final state -> [B,512]
-> logits [B,2]
```

再画短句补零后仍通过 LSTM cell 的时间轴。

## 代码走读

1. `train.py` 的 `build_vocab`：`<pad>=0`、`<unk>=1`、top-k。
2. `encode`：指出只保留前 200 token。
3. `model.py`：Embedding 的 `padding_idx=0`。
4. 解释双向两层 LSTM 的 `hn` 排列及 `hn[-2:]`。
5. 指出只改 `bidirectional=False` 会连锁影响状态提取与分类头维度。
6. 保存部分：权重和 vocab 为什么都需要。

## 现场演示

```bash
python 03_rnn_imdb/model.py
```

IMDb 下载和完整训练依赖网络，课前预缓存。课堂展示一个短文本和长文本编码后的 ID、有效长度与 pad 数，比直接等待训练更有教学价值。

## 必讲误区

- Embedding 表不是预先存在的词典释义，它会随任务训练。
- pad embedding 为零不等于 recurrent state 停止更新。
- 双向 LSTM 利用未来上下文，适合整句分类，不适合直接自回归生成。
- 当前 test 前 2000 条被当成 validation，没有独立最终 test。
- 分词、最大长度和标签映射都是部署契约的一部分。

## 课堂实验

比较前 200 token 与“前 100 + 后 100”策略。要求统计被截断评论比例，并分析影评结尾是否包含更强情感结论。

进阶实验是实现 `pack_padded_sequence`，比较短文本推理结果和训练时间。

## 提问

1. 为什么 token ID 不能直接作为连续数值输入？
2. `padding_idx=0` 具体保证了什么，没有保证什么？
3. 部署时只有 `.pt` 和 vocab 是否足够？还缺什么？

## 收束

RNN 通过递归携带历史，但难以并行且长距离路径较长。下一节使用注意力让任意 token 直接交互。
