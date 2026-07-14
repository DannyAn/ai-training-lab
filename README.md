# AI Training Lab 🧠

从零开始动手实现深度学习的经典模型和现代 AI 技术。每个模块独立可运行，循序渐进，覆盖计算机视觉、自然语言处理、大语言模型微调及 RAG 系统。

## 项目结构

| 模块 | 主题 | 难度 | 说明 |
|------|------|------|------|
| [01_mnist](./01_mnist) | CNN 手写数字识别 | ⭐ | 第一个神经网络：卷积、池化、全连接 |
| [02_cifar10](./02_cifar10) | CIFAR-10 图像分类 | ⭐⭐ | 数据增强、BatchNorm、Dropout |
| [03_rnn_imdb](./03_rnn_imdb) | RNN 情感分析 | ⭐⭐ | Embedding、LSTM/GRU、词向量 |
| [04_transformer_from_scratch](./04_transformer_from_scratch) | 手写 Transformer | ⭐⭐⭐ | 多头注意力、位置编码、编码器-解码器 |
| [05_gpt_tiny](./05_gpt_tiny) | 微型 GPT 语言模型 | ⭐⭐⭐ | 因果注意力、文本生成、自回归 |
| [06_bert_finetune](./06_bert_finetune) | BERT 微调 | ⭐⭐ | 预训练模型、迁移学习、HuggingFace |
| [07_lora_qwen](./07_lora_qwen) | Qwen LoRA 微调 | ⭐⭐⭐ | PEFT、QLoRA、指令微调 |
| [08_rag_demo](./08_rag_demo) | RAG 检索增强生成 | ⭐⭐⭐ | 向量数据库、Embedding、文档问答 |

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

# 安装依赖
pip install -r common/requirements.txt
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
2. **02_cifar10** — 掌握训练技巧（正则化、数据增强、学习率调度）
3. **03_rnn_imdb** — 学习序列模型（词嵌入、RNN/LSTM、文本分类）
4. **04_transformer_from_scratch** — 亲手实现 Transformer（注意力机制、位置编码）
5. **05_gpt_tiny** — 理解自回归语言模型（因果注意力、文本生成）
6. **06_bert_finetune** — 学会使用预训练模型做迁移学习
7. **07_lora_qwen** — 掌握参数高效微调（LoRA、QLoRA）
8. **08_rag_demo** — 构建 RAG 系统（向量检索、大模型问答）

## 技术栈

- **框架**: PyTorch
- **视觉**: torchvision
- **NLP**: HuggingFace Transformers, Tokenizers
- **微调**: PEFT, TRL
- **RAG**: LangChain, ChromaDB
- **工具**: tqdm, TensorBoard, wandb（可选）

## 许可证

MIT
