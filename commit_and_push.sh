#!/bin/bash
set -e  # 遇到错误立即退出

echo "=== 开始小批量提交 ==="

# 批次 1：项目文档和配置
echo -e "\n[1/7] 提交项目文档和配置..."
git add CLAUDE.md DEVELOPER.md .gitignore
git commit -m "docs: 添加 CLAUDE.md 和完善 DEVELOPER.md 学习路径

- 新增 CLAUDE.md：为 AI 助手提供项目结构和工作指引
- DEVELOPER.md：补充推荐学习项目列表和完整学习方案
- .gitignore：添加 data/ 和 ai-env/ 目录"

# 批次 2：共享工具改进
echo -e "\n[2/7] 提交共享工具改进..."
git add common/data.py
git commit -m "refactor(common): 统一数据集路径到 data/ 根目录

- 添加 DATA_ROOT 全局变量，锚定到仓库根目录
- 实现数据存在性检查，避免重复下载
- 所有模块的数据将集中存放，不会因运行目录不同而散落"

# 批次 3：基础视觉模块
echo -e "\n[3/7] 提交基础视觉模块注释增强..."
git add 01_mnist/train.py 02_cifar10/train.py
git commit -m "docs(vision): 为 MNIST 和 CIFAR10 模块添加详细学习注释

- 01_mnist: 添加【记忆】【调参】【产物】【使用】标注体系
- 02_cifar10: 同步注释风格，补充参数说明
- 补充模型保存和加载示例代码"

# 批次 4：NLP 基础模块
echo -e "\n[4/7] 提交 NLP 基础模块..."
git add 03_rnn_imdb/train.py
git commit -m "docs(nlp): 完善 IMDB 情感分析模块注释

- 添加详细的学习目标和调参说明
- 补充 RNN/LSTM 相关概念注释"

# 批次 5：Transformer 核心模块
echo -e "\n[5/7] 提交 Transformer 核心模块..."
git add 04_transformer_from_scratch/model.py 04_transformer_from_scratch/train.py 05_gpt_tiny/train.py 05_gpt_tiny/char_vocab.json
git commit -m "fix(transformer): 修复因果掩码逻辑并完善注释

- 04_transformer: 修复 generate_square_subsequent_mask 的掩码约定
  原实现使用 -inf 上三角，与 masked_fill(mask==0) 冲突导致 NaN
  改为返回布尔下三角矩阵 (True=可见, False=屏蔽)
- 05_gpt_tiny: 添加详细训练流程注释，新增 char_vocab.json
- 两个模块都补充了自回归训练的核心概念说明"

# 批次 6：预训练模型微调
echo -e "\n[6/7] 提交预训练模型微调模块..."
git add 06_bert_finetune/train.py 07_lora_qwen/train.py
git commit -m "docs(finetune): 完善 BERT 和 LoRA 微调模块注释

- 06_bert: 补充预训练模型微调的核心概念
- 07_lora: 详细说明参数高效微调(PEFT)原理
- 统一注释风格和学习标注"

# 批次 7：RAG 演示
echo -e "\n[7/7] 提交 RAG 演示模块..."
git add 08_rag_demo/train.py
git commit -m "docs(rag): 完善 RAG 检索增强生成模块注释

- 添加 RAG 工作流程和核心概念说明
- 补充向量数据库和检索器的使用说明"

echo -e "\n✅ 所有 7 个批次提交完成"
echo -e "\n=== 创建 GitHub 仓库并推送 ==="

# 创建 GitHub 仓库
echo -e "\n创建 GitHub 仓库 ai-training-lab..."
gh repo create ai-training-lab --public --source=. --description="从零实现深度学习：8个渐进式实战模块（MNIST→CIFAR10→IMDB→Transformer→GPT→BERT→LoRA→RAG）"

# 推送到 GitHub
echo -e "\n推送所有内容到 GitHub..."
git push -u origin main

echo -e "\n🎉 完成！仓库已创建并推送到 GitHub"
echo "访问地址: https://github.com/$(git config user.name)/ai-training-lab"
