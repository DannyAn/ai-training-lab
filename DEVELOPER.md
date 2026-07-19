

## 安装工具链

我建议安装这一整套
建议建立一个独立的 Python 环境，例如使用 venv 或 conda：
```
python -m venv ai-env
source ai-env/bin/activate
```

然后安装：

```
pip install torch torchvision torchaudio
```

然后：

```
pip install transformers
```

然后：

```
pip install datasets
```

然后：

```
pip install accelerate
```

然后：

```
pip install peft
```

然后：

```
pip install trl
```

然后：

```
pip install tensorboard
```

然后：

```
pip install jupyterlab
```

最后：

```
pip install matplotlib pandas scikit-learn
```

这一套已经覆盖了绝大多数学习和实验需求。


## 推荐的学习项目（按顺序）

不要看视频。

不要看几百小时课程。

一定要：

一个项目一个项目完成。


| 项目 | 学会什么 | 面试价值 | 说明 |
| :--- | :--- | :--- | :--- |
| MNIST 分类 | Dataset、DataLoader、训练循环 | ⭐⭐⭐⭐⭐ | 美国国家标准技术研究院发布的7万张黑白数据集，它是机器学习（尤其是计算机视觉）领域最经典、最基础的手写数字数据集。计算机视觉领域的 "Hello World" |
| CIFAR10 分类 | CNN、卷积、Pooling | ⭐⭐⭐⭐⭐ | 加拿大高级研究所发布的经典图像分类数据集，6万张彩色图像（10个分类，如飞机、汽车等）|
| IMDB 情感分析 | Embedding、LSTM、Tokenizer | ⭐⭐⭐⭐ | 自然语言处理（NLP）领域最经典的入门级二分类数据集，包含5万条来自 IMDB 网站的真实英文电影评论 |
| Tiny Shakespeare GPT | Transformer、Attention、自回归训练 | ⭐⭐⭐⭐⭐ | 经典的一个文本生成项目，是 Andrej Karpathy（前 Tesla AI 总监）整理的莎士比亚全集文本（约 1MB 大小） |
| BERT 文本分类 | Fine-tuning、预训练模型 | ⭐⭐⭐⭐⭐ | 自然语言处理（NLP）领域从“从零训练”跨越到“站在巨人肩膀上”的标志性项目。学会掌握“预训练+微调”的范式。 |
| LoRA 微调 Gemma/Qwen | PEFT、Adapter、参数高效微调 | ⭐⭐⭐⭐⭐ | 从“全量微调”跨越到“参数高效微调（PEFT）”的核心进阶项目。PEFT（参数高效微调）是目前大模型微调的行业标准。 |
| Tiny ChatBot | 推理、Tokenizer、生成流程 | ⭐⭐⭐⭐ | 将“模型训练”跨越到“模型推理与工程落地”的关键实战项目。 |
| RAG Demo | 检索增强 | ⭐⭐⭐ | RAG 是 Retrieval-Augmented Generation 的英文缩写，中文通常翻译为检索增强生成。 |





## 完整学习方案

你有二十多年的软件开发和安全经验，因此我不会建议你停留在“会训练一个模型”。你的优势应该是构建对整个训练系统的理解。

建议按下面四个阶段推进：

### PyTorch 基础（2 周）

PyTorch 基础（2 周）：Tensor、Autograd、Dataset/DataLoader、训练循环、CNN、MLP。

### Transformer 原理与实现（3～4 周）

从零实现一个 TinyGPT（如 Tiny Shakespeare），理解每个模块而不是只会调用库。

现代 LLM 微调（2～3 周）：学习 Hugging Face、Trainer、Accelerate、PEFT、LoRA，在 M1 上微调 0.5B 级别模型。
训练工程化（2～3 周）：TensorBoard、Checkpoint、配置管理、实验记录、混合精度、分布式训练原理（DDP/FSDP/DeepSpeed），即使本机不能运行，也要理解设计思想。

最终成果应该不是一堆 Notebook，而是一个自己的 GitHub 仓库，例如：

ai-training-lab/
├── 01_mnist/
├── 02_cifar10/
├── 03_rnn_imdb/
├── 04_transformer_from_scratch/
├── 05_gpt_tiny/
├── 06_bert_finetune/
├── 07_lora_qwen/
├── 08_rag_demo/
├── common/
└── README.md
