# 讲师与助教检查清单

## 一周前

- [ ] 确认授课形式：2 天集中课或 8 次模块课。
- [ ] 确认学员 Python、PyTorch、数学基础和操作系统分布。
- [ ] 确认机房/学员设备是否允许访问 PyPI、Hugging Face 和数据源。
- [ ] 决定 07 LoRA 是现场 GPU 演示、远程 GPU、预录，还是纯代码走读。
- [ ] 为长训练准备可复现日志、checkpoint 截图和 2-5 分钟预录视频。
- [ ] 通读根 `COURSE.md`、各模块 `COURSE.md` 和 `SLIDES.md`。
- [ ] 检查代码与课件中的模型名、参数和产物路径一致。

## 一天前：环境

- [ ] 从仓库根目录激活 `ai-env`。
- [ ] 执行 `python -m pip check`。
- [ ] 确认 `python -c "import torch; print(torch.__version__)"` 成功。
- [ ] 确认 CUDA/MPS/CPU 实际设备和可用内存。
- [ ] 预下载 MNIST、CIFAR-10、IMDb、SST-2。
- [ ] 预缓存 BERT、Qwen、MiniLM；需要 DeepSeek 演示时也提前缓存。
- [ ] 确认磁盘空间足够容纳模型 cache、checkpoint 和 Chroma。
- [ ] 不在课堂依赖第一次下载大模型。

## 一天前：逐模块烟雾测试

```bash
python 01_mnist/model.py
python 02_cifar10/model.py
python 03_rnn_imdb/model.py
python 04_transformer_from_scratch/model.py
python 05_gpt_tiny/model.py
python 06_bert_finetune/model.py
python 08_rag_demo/model.py
python 08_rag_demo/train.py
python 09_ai_agent/model.py
python 09_ai_agent/train.py --sim
```

- [ ] 对需要完整训练的模块至少跑通一个缩短副本。
- [ ] 检查终端字号、代码字体和投影可读性。
- [ ] 检查 `08_rag_demo/chroma_db` 可删除重建。
- [ ] 检查 07 的 Transformers/PEFT 参数是否因版本变化产生弃用错误。
- [ ] 检查 `09_ai_agent/train.py --sim` 模拟 Agent Loop 可正常运行。
- [ ] 如需演示 `--live`，提前缓存 Qwen2.5-1.5B-Instruct（约 3GB）。
- [ ] 将关键预期输出保存为纯文本，网络或 GPU 失败时可展示。

## 课堂文件准备

- [ ] PPTX/PDF 从当前 `SLIDES.md` 生成，没有脱离仓库维护的旧版本。
- [ ] 每个模块打开 `model.py`、`train.py`、`COURSE.md` 三个标签页。
- [ ] 准备架构图空白版，让学员现场补 shape/mask。
- [ ] 准备实验记录模板。
- [ ] 准备包含预期错误的示例：错误 transform、pad 未 mask、库外 RAG 问题。

## 每节课开始前

- [ ] 说明本节输入、输出、loss/评价指标和产物。
- [ ] 说明现场命令预计耗时、下载和硬件要求。
- [ ] 先提问题，再展示答案和代码。
- [ ] 先画数据流和 shape，再进入实现细节。

## 每节课结束前

- [ ] 明确说出当前 Demo 不能证明什么。
- [ ] 完成三个出门测问题。
- [ ] 给出只改变一个变量的实验。
- [ ] 说明下一模块为什么自然出现。

## 故障备份

- [ ] 网络失败：使用 cache、日志、截图或预录，不现场修改镜像配置。
- [ ] GPU OOM：使用预设小 batch/短序列副本，不随意改变多个参数。
- [ ] MPS 算子失败：切 CPU 完成 shape 演示，完整指标展示预录结果。
- [ ] 07 无 CUDA：坚持代码走读，不把 macOS 上安装 bitsandbytes 当课堂目标。
- [ ] RAG LLM 加载失败：继续完成纯 retrieval，强调检索可独立评估。

## 课后

- [ ] 收集学员最常见的三个误解，更新对应模块 `COURSE.md`。
- [ ] 保存实验配置与结果，不提交大型权重或数据库产物。
- [ ] 记录实际授课时长，调整下一期节奏。
- [ ] 如果代码有变化，按 `COURSE.md` 的更新顺序同步文档和幻灯片。
- [ ] 归档本期 PPTX/PDF，但继续以 Markdown 为内容源。
