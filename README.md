# AI Training Lab 🧠

从零开始动手实现深度学习的经典模型和现代 AI 技术。每个模块独立可运行，循序渐进，覆盖计算机视觉、自然语言处理、大语言模型微调及 RAG 系统。

## 课程资料

本仓库同时可以作为一套完整培训课程使用：

| 文档 | 用途 |
|---|---|
| [COURSE.md](./COURSE.md) | 整体课程目标、两天课表、教学主线和考核设计 |
| 各模块 `COURSE.md` | 讲师逐节口播、板书、代码演示、提问和课堂实验 |
| 各模块 `README.md` | 学员预习、复习、扩展知识和生产边界 |
| [SLIDES.md](./SLIDES.md) | 制作或导出培训 PPTX/PDF 的统一幻灯片大纲 |
| [AI_Training_Lab_Course_v1.pptx](./AI_Training_Lab_Course_v1.pptx) | 33 页第一版可编辑培训课件 |
| [INSTRUCTOR_CHECKLIST.md](./INSTRUCTOR_CHECKLIST.md) | 讲师和助教的环境、缓存、演示与故障备份清单 |

推荐先以 Markdown 维护课程内容，内容稳定后再从 `SLIDES.md` 制作 PPTX，避免代码、讲稿和二进制课件长期不同步。

## 项目结构

| 模块 | 主题 | 难度 | 说明 |
|------|------|------|------|
| [01_mnist](./01_mnist/README.md) ([讲师脚本](./01_mnist/COURSE.md)) | CNN 手写数字识别 | ⭐ | 第一个神经网络：卷积、池化、全连接 |
| [02_cifar10](./02_cifar10/README.md) ([讲师脚本](./02_cifar10/COURSE.md)) | CIFAR-10 图像分类 | ⭐⭐ | 数据增强、BatchNorm、Dropout |
| [03_rnn_imdb](./03_rnn_imdb/README.md) ([讲师脚本](./03_rnn_imdb/COURSE.md)) | RNN 情感分析 | ⭐⭐ | Embedding、LSTM、词向量 |
| [04_transformer_from_scratch](./04_transformer_from_scratch/README.md) ([讲师脚本](./04_transformer_from_scratch/COURSE.md)) | 手写 Transformer | ⭐⭐⭐ | 多头注意力、位置编码、编码器-解码器 |
| [05_gpt_tiny](./05_gpt_tiny/README.md) ([讲师脚本](./05_gpt_tiny/COURSE.md)) | 微型 GPT 语言模型 | ⭐⭐⭐ | 因果注意力、文本生成、自回归 |
| [06_bert_finetune](./06_bert_finetune/README.md) ([讲师脚本](./06_bert_finetune/COURSE.md)) | BERT 微调 | ⭐⭐ | 预训练模型、迁移学习、HuggingFace |
| [07_lora_qwen](./07_lora_qwen/README.md) ([讲师脚本](./07_lora_qwen/COURSE.md)) | Qwen LoRA 微调 | ⭐⭐⭐ | PEFT、QLoRA、指令微调 |
| [08_rag_demo](./08_rag_demo/README.md) ([讲师脚本](./08_rag_demo/COURSE.md)) | RAG 检索增强生成 | ⭐⭐⭐ | 向量数据库、Embedding、文档问答 |
| [09_ai_agent](./09_ai_agent/README.md) ([讲师脚本](./09_ai_agent/COURSE.md)) | AI Agent 实战 | ⭐⭐⭐ | Agent Loop、Function Calling、Tool/Skill |

每个模块的 `README.md` 不只是运行说明，还记录了代码中不容易直接看到的知识：张量和数据流、关键设计选择、教学实现与生产实现的边界、常见误区以及建议实验。推荐按下面的顺序学习一个模块：

1. 先读模块 `README.md` 的“模型/数据流”，明确每个组件的职责。
2. 运行 `model.py`，检查输入输出 shape 和参数量。
3. 对照注释阅读 `train.py`，再完整运行一次。
4. 回到模块 `README.md` 阅读“教学版边界”，避免把 Demo 行为误认为标准生产方案。
5. 从“建议实验”中选择一个变量做对照实验，并记录配置、指标和结论。

## 开始使用

### 环境要求

- Python >= 3.10
- PyTorch >= 2.0
- 可选 GPU（CUDA 支持）

### 安装

```bash
# 克隆项目
git clone https://github.com/<your-username>/ai-training-lab.git
cd ai-training-lab

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖（根 requirements.txt 是完整、较新的依赖清单）
pip install -r requirements.txt
```

### 运行示例

```bash
# 训练 MNIST 分类器
python 01_mnist/train.py

# 训练 CIFAR-10 分类器
python 02_cifar10/train.py

# 训练 RNN 情感分析
python 03_rnn_imdb/train.py

# 训练 Transformer
python 04_transformer_from_scratch/train.py
```

每个模块都有独立的 `train.py`，可以用 `python <module>/train.py` 直接运行。

## 学习路线

如果你刚开始学习深度学习，建议按编号顺序学习：

1. **01_mnist** — 理解神经网络的基本组件（卷积、激活函数、反向传播）
2. **02_cifar10** — 掌握训练技巧（正则化、数据增强、BatchNorm）
3. **03_rnn_imdb** — 学习序列模型（词嵌入、RNN/LSTM、文本分类）
4. **04_transformer_from_scratch** — 亲手实现 Transformer（注意力机制、位置编码）
5. **05_gpt_tiny** — 理解自回归语言模型（因果注意力、文本生成）
6. **06_bert_finetune** — 学会使用预训练模型做迁移学习
7. **07_lora_qwen** — 掌握参数高效微调（LoRA、QLoRA）
8. **08_rag_demo** — 构建 RAG 系统（向量检索、大模型问答）
9. **09_ai_agent** — 构建 AI Agent（决策循环、Function Calling、Tool/Skill）

## 技术栈

- **框架**: PyTorch
- **视觉**: torchvision
- **NLP**: HuggingFace Transformers, Tokenizers
- **微调**: PEFT, TRL
- **RAG**: LangChain, ChromaDB
- **工具**: tqdm, TensorBoard, wandb（可选）

## 许可证

MIT
