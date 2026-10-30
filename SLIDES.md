# AI Training Lab 幻灯片大纲

本文是未来制作 PPTX/PDF 的内容源。每个 `##` 对应一页或一组紧密相连的幻灯片。幻灯片只放视觉主线和关键词；详细口播以各模块 `COURSE.md` 为准。

当前第一版 PPTX 是基于本大纲压缩设计的 33 页演示稿，可通过以下命令重新生成：

```bash
pip install -r tools/requirements.txt
python tools/generate_course_pptx.py
```

生成文件：`AI_Training_Lab_Course_v1.pptx`。生成脚本使用 PowerPoint 原生形状和文本，课件可在 PowerPoint 或 Keynote 中继续编辑。

## 01 封面

从神经网络训练到大模型应用实践

副标题：CNN、RNN、Transformer、GPT、BERT、LoRA 与 RAG

## 02 为什么学习这条路线

```text
会调用 API
    ↓
理解数据、模型、训练和评估
    ↓
能定位问题、选择方案、判断边界
```

## 03 课程地图

```text
MNIST -> CIFAR-10 -> RNN -> Transformer
-> GPT -> BERT -> LoRA -> RAG
```

## 04 四类任务

- 分类：输入 -> 类别
- 序列分类：token 序列 -> 类别
- 生成：前缀 -> 下一个 token
- RAG：问题 -> 检索资料 -> 生成答案

## 05 所有训练共享的闭环

```text
batch -> forward -> loss -> backward -> optimizer.step
                    ↓
                 validation
                    ↓
                 checkpoint
```

## 06 指标词典

- logits：未归一化分数
- probability：归一化概率
- loss：优化目标
- metric：任务评价
- train/validation/test：不同职责

## 07 MNIST 问题场景

一张手写数字如何变成十个分类分数？

## 08 CNN shape 图

```text
1x28x28 -> 32x14x14 -> 64x7x7 -> 3136 -> 128 -> 10
```

## 09 一个 batch 的训练

四步动画：forward、loss、backward、step。

## 10 checkpoint 里有什么

有：parameter、buffer。

没有：Python 类、数据预处理、业务接口、完整训练上下文。

## 11 MNIST 的隐藏事实

约 95% 参数集中在第一个全连接层；validation 不是官方 test；best 指标不等于 best checkpoint。

## 12 CIFAR-10 为什么更难

颜色、背景、视角、类内差异、真实世界变化。

## 13 更深 CNN

空间尺寸下降、通道上升、全局平均池化收束。

## 14 三种正则化/稳定化

- 数据增强：改变输入
- Dropout：改变训练激活
- weight decay：约束参数
- BatchNorm：规范激活并维护统计状态

## 15 验证 transform 陷阱

一张图说明 `Subset` 保存索引但共享底层 Dataset transform。

## 16 模型选择协议

```text
train 更新参数
validation 选配置/checkpoint
test 最终一次报告
```

## 17 文本为什么需要 tokenization

字符串 -> token -> ID；ID 只是索引，不包含大小关系。

## 18 RNN 数据流

```text
text -> vocab -> IDs -> Embedding -> BiLSTM -> logits
```

## 19 Padding 不等于忽略

对比：零 embedding 仍通过 recurrent cell；packed sequence 真正跳过补齐。

## 20 双向上下文

分类可看左右；自回归生成不能看未来。

## 21 预处理也是模型契约

tokenizer、vocab、max length、special IDs、label mapping。

## 22 为什么需要 Attention

RNN 长路径与难并行，对比任意 token 的直接连接。

## 23 Attention 公式

```text
softmax(QK^T / sqrt(d_k)) V
```

## 24 Multi-head

固定 d_model 下拆分多个表示子空间；head 数不等于参数量同比增加。

## 25 三种注意力

Encoder self-attention、Decoder causal self-attention、Cross-attention。

## 26 两种 mask

causal mask 与 padding mask 并排展示；loss ignore 单独列出。

## 27 Teacher forcing

```text
decoder input: target[:-1]
label:         target[1:]
```

## 28 Toy 数据边界

独立随机 source/target 不能证明翻译能力；随机目标理论 loss 约 `ln(48)`。

## 29 从 Transformer 到 GPT

删除 encoder 和 cross-attention，保留 causal decoder stack。

## 30 GPT 训练目标

每个位置预测下一个字符/token。

## 31 字符级词表与窗口

滑动窗口高度重叠；先切连续文本，再构造 train/validation 窗口。

## 32 Causal LM 并行训练

训练一次预测所有位置，mask 阻止答案泄漏。

## 33 自回归生成

最后位置 logits -> temperature/top-k -> sample -> append。

## 34 KV Cache

无 cache 重算历史；有 cache 复用历史 K/V。

## 35 Perplexity 的单位

字符、byte、BPE 的 perplexity 不可直接比较。

## 36 从零训练为什么昂贵

数据、算力、稳定优化、评估与安全；引出预训练迁移。

## 37 BERT 预训练微调

通用语言表示 + 小型下游任务数据。

## 38 BERT 输入

`[CLS] tokens [SEP] [PAD]`、input IDs、attention mask。

## 39 BERT 与 GPT

双向编码 vs 因果生成；任务目标决定 mask。

## 40 Pooling 选择

pooler output、raw CLS、masked mean。

## 41 全量微调与冻结

参数量、显存、速度、适配能力和学习率的权衡。

## 42 动态 Padding

固定 128 与 batch longest 的计算浪费对比。

## 43 为什么需要 PEFT

每个任务复制完整大模型的训练、存储和部署成本。

## 44 LoRA 公式

```text
W' = W + (alpha/r) BA
```

## 45 LoRA 参数

r、alpha、dropout、target modules。

## 46 LoRA vs QLoRA

冻结基座相同；基座加载精度和权重显存不同。

## 47 Adapter 产物

adapter 不是完整模型；依赖 base、revision、tokenizer、template。

## 48 SFT 数据与 loss mask

全序列监督 vs assistant-only 监督。

## 49 微调评估泄漏

三条样本复制后随机切分，不等于独立验证。

## 50 为什么事实知识不总适合微调

更新频率、来源引用、删除与权限；引出 RAG。

## 51 RAG 三组件

Embedding 模型、向量数据库、生成 LLM，以三种颜色展示。

## 52 RAG 离线建库

文档 -> chunk -> embedding -> Chroma。

## 53 RAG 在线问答

问题 -> query vector -> top-k chunks -> prompt -> LLM。

## 54 Embedding 不是数据库

文本编码器输出向量；Chroma 保存数据；LLM 生成语言。

## 55 同一向量空间

文档和问题必须使用同一模型/版本/设置；同维度不等于同空间。

## 56 ChromaDB 的定位

产品支持本地、server、cloud；本课使用本地嵌入式教学模式。

## 57 Retrieval 与 Generation 分评

Recall@k/MRR 与忠实性/完整性/引用准确度。

## 58 top-k 不等于相关

库外问题仍会返回“最接近”内容；threshold、拒答和 reranker。

## 59 RAG 安全

权限过滤、敏感数据、间接 prompt injection、引用验证。

## 60 LoRA 与 RAG 决策

| 目标 | 方法 |
|---|---|
| 更新事实与引用 | RAG |
| 改变格式/语气/行为 | LoRA |
| 二者同时需要 | 组合 |

## 61 从 RAG 到 Agent

RAG 给了模型”外部记忆”，Agent 进一步给了模型”手和决策权”。

```text
RAG: 用户问题 → 检索 → LLM → 答案     （被动：每次必检索）
Agent: 用户任务 → LLM 决定 → 调用工具 → 观察结果 → 再思考 → 答案  （自主：自己决定做什么）
```

## 62 Agent Loop 四阶段

```text
Thought(思考) → Action(调用工具) → Observation(观察结果) → 循环或 Final Answer(最终回答)
```

每一步历史追加到下一轮 prompt，模型基于”自己之前的决策和结果”调整行为。

## 63 Agent 的核心不是模型，是循环

三种颜色标注：模型推理（蓝）、工具执行（紫）、循环控制（绿）。

Agent = LLM + 工具集 + 决策循环 + 终止条件（max_steps），不是新的模型架构。

## 64 Function Calling 的实现方式

ReAct 模式：工具描述注入 system prompt → 模型输出 Thought/Action/Action Input 文本 → 正则解析 → 执行 Python 函数 → 返回 Observation。

工具描述是 prompt engineering 的一部分——描述模糊，模型就选错工具。

## 65 Tool 的四个要素

```text
Tool(name=”calculator”, description=”执行数学计算”, parameters=”expression: 数学表达式”, func=_calculator)
```

name 让模型选择工具，description 和 parameters 告诉模型如何使用，func 实际执行。工具执行在模型外部，模型只看到返回的字符串。

## 66 课程能力路线

```text
训练(01-04) → 从零训练语言模型(05) → 预训练微调(06)
→ PEFT/LoRA(07) → RAG 外部知识(08) → Agent 自主决策(09)
```

Agent 不是终点：模型、工具、记忆、规划、多 Agent 协作的工程组合。

## 67 教学 Demo 与生产系统

数据质量、独立评估、版本锁定、可恢复产物、服务化、监控、安全。

## 68 结课设计题

给出一个企业知识助手场景，让学员选择分类器、BERT、LoRA、RAG、Agent 或组合，并说明评估方法。

## 69 学习不是”跑通”

```text
提出假设 -> 控制变量 -> 记录指标 -> 解释结果 -> 说明限制
```

## 70 下一步

完成一个结课项目；把教学代码改为有独立评估、可恢复产物和明确边界的工程版本。

## 视觉制作规范

- 每页只讲一个结论，正文不超过六行。
- Shape、数据流和职责优先画图，不贴大段代码。
- 代码截图最多 8-12 行，并标出对应文件。
- 教学实现问题使用“当前代码行为”，避免表述为框架缺陷。
- 统一颜色：数据为蓝色、模型为紫色、数据库/产物为绿色、风险为红色。
- 页脚标注模块和文档路径，便于学员回到仓库。
