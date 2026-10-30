"""生成 AI Training Lab 第一版培训课件。"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "AI_Training_Lab_Course_v1.pptx"

W, H = 13.333, 7.5
BG = "F6F2E8"
NAVY = "102A43"
INK = "243B53"
MUTED = "627D98"
TEAL = "00A6A6"
CORAL = "F06449"
GOLD = "F2B134"
GREEN = "3A7D44"
WHITE = "FFFFFF"
PALE_BLUE = "E5F4F4"
PALE_CORAL = "FBE9E4"
PALE_GOLD = "FFF2CE"


def rgb(value):
    return RGBColor.from_string(value)


def set_bg(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = rgb(color)


def add_text(slide, text, x, y, w, h, size=22, color=INK, bold=False,
             font="Hiragino Sans GB", align=PP_ALIGN.LEFT,
             valign=MSO_ANCHOR.MIDDLE, margin=0.06):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_rect(slide, x, y, w, h, fill, line=None, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line or fill)
    return shape


def add_line(slide, x1, y1, x2, y2, color=TEAL, width=2.5):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def title(slide, index, section, heading, subtitle=None, dark=False):
    fg = WHITE if dark else NAVY
    muted = "B8CEDD" if dark else MUTED
    add_text(slide, f"{index:02d}", 0.55, 0.35, 0.6, 0.35, 13, TEAL, True)
    add_text(slide, section.upper(), 1.18, 0.35, 2.8, 0.35, 11, muted, True)
    add_text(slide, heading, 0.55, 0.82, 12.1, 0.72, 29, fg, True)
    if subtitle:
        add_text(slide, subtitle, 0.58, 1.48, 11.8, 0.45, 14, muted)


def footer(slide, index, label):
    add_text(slide, label, 0.58, 7.12, 5.5, 0.23, 9, MUTED)
    add_text(slide, str(index), 12.25, 7.12, 0.5, 0.23, 9, MUTED, align=PP_ALIGN.RIGHT)


def node(slide, text, x, y, w, h, fill=WHITE, accent=TEAL, size=17):
    add_rect(slide, x, y, w, h, fill, accent)
    add_text(slide, text, x + 0.08, y + 0.06, w - 0.16, h - 0.12, size, INK, True,
             align=PP_ALIGN.CENTER)


def flow(slide, labels, y=3.25, fills=None, sizes=None):
    n = len(labels)
    gap = 0.28
    total = 11.9
    box_w = (total - gap * (n - 1)) / n
    x = 0.72
    for i, label in enumerate(labels):
        fill = fills[i] if fills else WHITE
        size = sizes[i] if sizes else 16
        node(slide, label, x, y, box_w, 1.05, fill, TEAL if i < n - 1 else CORAL, size)
        if i < n - 1:
            add_line(slide, x + box_w, y + 0.52, x + box_w + gap - 0.03, y + 0.52)
        x += box_w + gap


def bullets(slide, items, x=0.85, y=2.1, w=11.7, h=4.5, size=22, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    for i, item in enumerate(items):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = item
        p.font.name = "Hiragino Sans GB"
        p.font.size = Pt(size)
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(15)
        p.level = 0
        p.text = "•  " + p.text
    return box


def two_cards(slide, left_title, left_items, right_title, right_items,
              left_color=TEAL, right_color=CORAL):
    for x, head, items, accent in [
        (0.72, left_title, left_items, left_color),
        (6.83, right_title, right_items, right_color),
    ]:
        add_rect(slide, x, 2.0, 5.78, 4.45, WHITE, accent)
        add_rect(slide, x, 2.0, 5.78, 0.72, accent, accent, False)
        add_text(slide, head, x + 0.25, 2.1, 5.25, 0.48, 19, WHITE, True)
        bullets(slide, items, x + 0.28, 2.92, 5.15, 3.2, 17)


def base_slide(prs, index, section, heading, subtitle=None, dark=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, NAVY if dark else BG)
    title(slide, index, section, heading, subtitle, dark)
    if not dark:
        footer(slide, index, "AI Training Lab · 课程课件 v1")
    return slide


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "AI Training Lab Course v1"
    prs.core_properties.subject = "从神经网络训练到大模型应用实践"
    prs.core_properties.author = "AI Training Lab"

    # 1 Cover
    s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, NAVY)
    add_rect(s, 0.0, 0.0, 0.22, H, TEAL, TEAL, False)
    add_text(s, "AI TRAINING LAB", 0.82, 0.72, 4.0, 0.4, 13, TEAL, True, font="Avenir Next")
    add_text(s, "从神经网络训练到\n大模型应用实践", 0.82, 1.55, 8.3, 1.75, 34, WHITE, True)
    add_text(s, "CNN · RNN · Transformer · GPT · BERT · LoRA · RAG", 0.88, 3.58, 9.8, 0.5, 17, "B8CEDD")
    for i, (label, c) in enumerate([("训练", TEAL), ("迁移", GOLD), ("应用", CORAL)]):
        x = 0.9 + i * 2.0
        add_rect(s, x, 5.25, 1.55, 0.6, c, c)
        add_text(s, label, x, 5.28, 1.55, 0.5, 16, NAVY if c == GOLD else WHITE, True, align=PP_ALIGN.CENTER)
    add_text(s, "第一版 · 讲师演示稿", 0.9, 6.62, 3.2, 0.32, 11, "8FAABD")

    # 2 Map
    s = base_slide(prs, 2, "Course Map", "一条从训练到底层应用的路线",
                   "每个模块都回答一个新的工程问题")
    flow(s, ["01\nMNIST", "02\nCIFAR", "03\nRNN", "04\nTransformer",
             "05\nGPT", "06\nBERT", "07\nLoRA", "08\nRAG"], y=3.05,
         fills=[PALE_BLUE, PALE_BLUE, PALE_GOLD, PALE_GOLD, PALE_CORAL, PALE_CORAL, PALE_BLUE, PALE_BLUE],
         sizes=[14] * 8)
    add_text(s, "图像训练", 0.75, 4.5, 2.7, 0.38, 14, TEAL, True, align=PP_ALIGN.CENTER)
    add_text(s, "序列与注意力", 3.55, 4.5, 2.75, 0.38, 14, GOLD, True, align=PP_ALIGN.CENTER)
    add_text(s, "预训练与生成", 6.45, 4.5, 2.75, 0.38, 14, CORAL, True, align=PP_ALIGN.CENTER)
    add_text(s, "大模型应用", 9.35, 4.5, 2.75, 0.38, 14, TEAL, True, align=PP_ALIGN.CENTER)

    # 3 Outcomes
    s = base_slide(prs, 3, "Outcomes", "课程结束，你应该能够……")
    two_cards(s, "读懂训练", ["沿 shape 追踪数据", "解释 loss 与 gradient", "识别 validation/test"],
              "做出选择", ["从零训练 vs 迁移", "LoRA vs RAG", "Demo vs 生产系统"])

    # 4 loop
    s = base_slide(prs, 4, "Foundation", "所有模型共享同一个训练闭环")
    flow(s, ["Batch\n数据", "Forward\n预测", "Loss\n误差", "Backward\n梯度", "Step\n更新"], y=2.75,
         fills=[PALE_BLUE, WHITE, PALE_CORAL, PALE_GOLD, PALE_BLUE])
    add_text(s, "Validation 不更新参数，只评价当前模型", 2.6, 4.55, 8.1, 0.65, 20, NAVY, True, align=PP_ALIGN.CENTER)
    add_line(s, 9.9, 3.8, 9.9, 4.45, CORAL)

    # 5 Metrics
    s = base_slide(prs, 5, "Foundation", "别把四个词混成一个")
    for i, (head, desc, c) in enumerate([
        ("LOGITS", "模型输出的未归一化分数", TEAL),
        ("PROBABILITY", "归一化后的解释形式", GOLD),
        ("LOSS", "优化器真正最小化的目标", CORAL),
        ("METRIC", "面向任务的评价指标", GREEN),
    ]):
        x = 0.72 + (i % 2) * 6.1; y = 2.05 + (i // 2) * 2.05
        add_rect(s, x, y, 5.75, 1.55, WHITE, c)
        add_text(s, head, x + 0.25, y + 0.15, 2.25, 0.4, 15, c, True, font="Avenir Next")
        add_text(s, desc, x + 0.25, y + 0.66, 5.2, 0.58, 18, INK, True)

    # 6 MNIST
    s = base_slide(prs, 6, "01 · MNIST", "第一条完整训练流水线", "像素如何变成十个类别分数？")
    flow(s, ["1×28×28", "Conv + Pool\n32×14×14", "Conv + Pool\n64×7×7", "Flatten\n3136", "Linear\n10 logits"], y=2.65)
    add_text(s, "约 95% 参数集中在第一个全连接层", 2.55, 4.45, 8.25, 0.62, 21, CORAL, True, align=PP_ALIGN.CENTER)

    # 7 checkpoint
    s = base_slide(prs, 7, "01 · MNIST", "Checkpoint 不是一个完整应用")
    two_cards(s, "state_dict 里有", ["模型参数", "BatchNorm 等 buffer", "可重新加载的张量"],
              "state_dict 里没有", ["Python 模型类", "预处理与标签含义", "完整训练配置"])

    # 8 CIFAR
    s = base_slide(prs, 8, "02 · CIFAR-10", "更难的数据，需要更强的泛化能力")
    flow(s, ["3×32×32", "64×16×16", "128×8×8", "256×4×4", "GAP\n256", "10 logits"], y=2.45)
    bullets(s, ["数据增强改变输入，而不是改变标签", "BatchNorm 在 train / eval 使用不同统计量", "全局平均池化压缩分类头"], y=4.1, h=2.2, size=19)

    # 9 validation trap
    s = base_slide(prs, 9, "02 · CIFAR-10", "一个真实的验证集陷阱", "Subset 保存索引，但共享底层 Dataset 的 transform")
    node(s, "CIFAR Dataset\ntrain transform", 0.9, 2.45, 3.0, 1.25, PALE_CORAL, CORAL)
    node(s, "Train Subset\n随机增强", 5.15, 2.0, 2.9, 1.05, PALE_BLUE, TEAL)
    node(s, "Val Subset\n也随机增强！", 5.15, 3.7, 2.9, 1.05, PALE_CORAL, CORAL)
    add_line(s, 3.9, 3.05, 5.08, 2.53); add_line(s, 3.9, 3.1, 5.08, 4.2, CORAL)
    add_text(s, "修正：相同索引划分，不同 Dataset transform", 8.45, 2.65, 3.65, 1.2, 19, NAVY, True, align=PP_ALIGN.CENTER)

    # 10 RNN
    s = base_slide(prs, 10, "03 · RNN", "文本先变成 ID，再变成表示")
    flow(s, ["Text", "Tokens", "Token IDs", "Embedding\n[B,T,128]", "BiLSTM", "2 logits"], y=2.55)
    add_text(s, "ID 是索引，不是语义数值", 3.15, 4.35, 7.0, 0.62, 22, CORAL, True, align=PP_ALIGN.CENTER)

    # 11 Padding
    s = base_slide(prs, 11, "03 · RNN", "padding_idx=0 不等于 LSTM 忽略 padding")
    for i, token in enumerate(["好", "看", "<PAD>", "<PAD>", "<PAD>"]):
        c = TEAL if i < 2 else MUTED
        node(s, token, 1.0 + i * 2.25, 2.3, 1.45, 0.8, WHITE, c, 14)
        if i < 4: add_line(s, 2.45 + i * 2.25, 2.7, 3.15 + i * 2.25, 2.7, c)
    add_text(s, "零向量仍会经过 recurrent weights 和 bias", 2.2, 3.65, 8.9, 0.65, 21, CORAL, True, align=PP_ALIGN.CENTER)
    add_text(s, "真正跳过：lengths + packed sequence，或 masked pooling", 1.6, 4.6, 10.1, 0.55, 18, NAVY, align=PP_ALIGN.CENTER)

    # 12 Attention
    s = base_slide(prs, 12, "04 · Transformer", "Attention 缩短任意 token 之间的路径")
    add_rect(s, 0.9, 2.15, 11.5, 1.2, NAVY, NAVY)
    add_text(s, "Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V", 1.05, 2.36, 11.2, 0.65, 25, WHITE, True, font="Menlo", align=PP_ALIGN.CENTER)
    bullets(s, ["Q：我在找什么", "K：我能被怎样匹配", "V：匹配后取走什么信息"], y=4.05, h=2.2, size=20)

    # 13 three attention
    s = base_slide(prs, 13, "04 · Transformer", "Encoder-Decoder 中的三种注意力")
    for x, head, desc, c in [
        (0.72, "Encoder Self", "source 内部互看", TEAL),
        (4.72, "Decoder Causal", "target 不能看未来", CORAL),
        (8.72, "Cross Attention", "target 查询 source", GOLD),
    ]:
        add_rect(s, x, 2.25, 3.72, 2.65, WHITE, c)
        add_text(s, head, x + 0.15, 2.52, 3.42, 0.48, 18, c, True, font="Avenir Next", align=PP_ALIGN.CENTER)
        add_text(s, desc, x + 0.3, 3.38, 3.12, 0.85, 20, INK, True, align=PP_ALIGN.CENTER)

    # 14 masks
    s = base_slide(prs, 14, "04 · Transformer", "三种“忽略”机制，三个不同职责")
    two_cards(s, "Attention 读取范围", ["Causal mask：屏蔽未来", "Padding mask：屏蔽补齐 key", "作用于 attention scores"],
              "Loss 计分范围", ["ignore_index：不计目标 pad", "不会阻止 attention 读取", "作用于监督信号"])

    # 15 toy data
    s = base_slide(prs, 15, "04 · Transformer", "Toy 数据跑通机制，不证明翻译能力", dark=True)
    add_text(s, "source 与 target 独立随机", 1.15, 2.25, 11.0, 0.65, 25, WHITE, True, align=PP_ALIGN.CENTER)
    add_text(s, "理论基线 loss ≈ ln(48) ≈ 3.87", 1.15, 3.4, 11.0, 0.7, 27, GOLD, True, font="Menlo", align=PP_ALIGN.CENTER)
    add_text(s, "loss 下降可能只是记忆固定样本", 1.15, 4.72, 11.0, 0.55, 19, "B8CEDD", align=PP_ALIGN.CENTER)

    # 16 GPT
    s = base_slide(prs, 16, "05 · GPT", "Decoder-only：统一为下一 token 预测")
    flow(s, ["Token IDs", "Token + Pos\nEmbedding", "Causal Blocks\n× N", "LM Head", "Next-token\nDistribution"], y=2.55)
    add_text(s, "训练：一次预测所有位置  ·  生成：逐 token 等待", 1.4, 4.45, 10.5, 0.62, 21, NAVY, True, align=PP_ALIGN.CENTER)

    # 17 sliding windows
    s = base_slide(prs, 17, "05 · GPT", "55936 个窗口，不等于 55936 条独立文本")
    add_text(s, "样本 i", 0.95, 2.2, 1.1, 0.4, 14, MUTED, True)
    node(s, "H e l l o   w o r l d ...", 2.0, 2.05, 8.8, 0.75, WHITE, TEAL, 18)
    add_text(s, "样本 i+1", 0.95, 3.45, 1.1, 0.4, 14, MUTED, True)
    node(s, "  e l l o   w o r l d ...", 2.25, 3.3, 8.8, 0.75, WHITE, CORAL, 18)
    add_text(s, "随机拆窗口会让 train / validation 严重泄漏", 1.45, 4.8, 10.5, 0.65, 21, CORAL, True, align=PP_ALIGN.CENTER)

    # 18 Sampling
    s = base_slide(prs, 18, "05 · GPT", "Sampling 参数改变输出分布，不改变模型知识")
    two_cards(s, "Temperature", ["低：更尖锐、更保守", "高：更多样、更易出错", "0 应改用显式 argmax"],
              "Top-k", ["只保留最高 k 个候选", "减少低概率异常 token", "应多次采样再评价"])

    # 19 KV cache
    s = base_slide(prs, 19, "05 · GPT", "KV Cache：不要反复计算已经看过的历史")
    node(s, "历史 tokens", 0.9, 2.5, 2.2, 1.0, PALE_BLUE, TEAL)
    node(s, "历史 K / V\n缓存", 4.15, 2.5, 2.2, 1.0, PALE_GOLD, GOLD)
    node(s, "只计算\n新 token", 7.4, 2.5, 2.2, 1.0, PALE_CORAL, CORAL)
    node(s, "输出下一 token", 10.35, 2.5, 2.2, 1.0, WHITE, TEAL)
    for a, b in [(3.1, 4.12), (6.35, 7.37), (9.6, 10.32)]: add_line(s, a, 3.0, b, 3.0)
    add_text(s, "训练追求并行 · 推理追求复用", 2.4, 4.55, 8.6, 0.62, 22, NAVY, True, align=PP_ALIGN.CENTER)

    # 20 BERT vs GPT
    s = base_slide(prs, 20, "06 · BERT", "同是 Transformer，任务目标决定注意力方向")
    two_cards(s, "BERT · 编码理解", ["双向上下文", "[CLS] / pooling", "分类、抽取、匹配"],
              "GPT · 自回归生成", ["只能看左侧", "next-token prediction", "续写、对话、生成"])

    # 21 Fine tuning
    s = base_slide(prs, 21, "06 · BERT", "微调：站在预训练表示上适配任务")
    flow(s, ["预训练 BERT\n≈ 1.1 亿参数", "SST-2\n任务数据", "全量微调\nlr=2e-5", "情感分类器"], y=2.65,
         fills=[PALE_BLUE, WHITE, PALE_CORAL, PALE_GOLD])
    add_text(s, "不是从随机参数重新学习语言", 2.35, 4.5, 8.7, 0.6, 21, CORAL, True, align=PP_ALIGN.CENTER)

    # 22 dynamic padding
    s = base_slide(prs, 22, "06 · BERT", "固定 Padding 很简单，但可能把算力花在 PAD 上")
    for row, n, c in [(0, 3, TEAL), (1, 6, GOLD), (2, 9, CORAL)]:
        y = 2.15 + row * 1.12
        for col in range(10):
            fill = c if col < n else "D9E2E8"
            add_rect(s, 1.0 + col * 1.1, y, 0.9, 0.68, fill, fill, False)
    add_text(s, "动态 padding：每个 batch 只补到当前最长序列", 2.0, 5.65, 9.2, 0.55, 20, NAVY, True, align=PP_ALIGN.CENTER)

    # 23 LoRA
    s = base_slide(prs, 23, "07 · LoRA", "冻结基座，只学习低秩参数增量")
    add_rect(s, 1.15, 2.15, 11.0, 1.2, NAVY, NAVY)
    add_text(s, "W′ = W + (α / r) · B A", 1.3, 2.38, 10.7, 0.65, 28, WHITE, True, font="Menlo", align=PP_ALIGN.CENTER)
    flow(s, ["W\n冻结", "A / B\n可训练", "Adapter\n按任务保存"], y=4.1,
         fills=[PALE_BLUE, PALE_CORAL, PALE_GOLD])

    # 24 LoRA vs QLoRA
    s = base_slide(prs, 24, "07 · LoRA", "LoRA 与 QLoRA：训练增量相似，基座精度不同")
    two_cards(s, "LoRA", ["基座 fp16 / bf16 / fp32", "基座冻结", "训练 adapter"],
              "QLoRA", ["基座 4-bit 加载", "进一步降低权重显存", "依赖 CUDA / bitsandbytes"])

    # 25 adapter
    s = base_slide(prs, 25, "07 · LoRA", "Adapter 不是完整模型")
    flow(s, ["Base Model\n+ revision", "Tokenizer\n+ chat template", "LoRA\nAdapter", "可复现推理"], y=2.65,
         fills=[PALE_BLUE, PALE_GOLD, PALE_CORAL, WHITE])
    add_text(s, "三条样本复制后随机切分 ≠ 独立评估", 2.0, 4.55, 9.3, 0.62, 21, CORAL, True, align=PP_ALIGN.CENTER)

    # 26 RAG components
    s = base_slide(prs, 26, "08 · RAG", "一个基础 RAG 是三个独立组件")
    for x, head, desc, c in [
        (0.72, "Embedding 模型", "文本 → 384 维向量\n不保存文档，不生成答案", TEAL),
        (4.72, "ChromaDB", "保存向量 + 原文 + metadata\n执行相似度检索", GREEN),
        (8.72, "生成 LLM", "问题 + 检索原文 → 答案\n默认 Qwen2.5-1.5B", CORAL),
    ]:
        add_rect(s, x, 2.05, 3.72, 3.35, WHITE, c)
        add_text(s, head, x + 0.18, 2.35, 3.36, 0.5, 20, c, True, align=PP_ALIGN.CENTER)
        add_text(s, desc, x + 0.25, 3.25, 3.22, 1.25, 17, INK, True, align=PP_ALIGN.CENTER)

    # 27 RAG flow
    s = base_slide(prs, 27, "08 · RAG", "离线建库与在线问答是两条流水线")
    add_text(s, "离线", 0.75, 2.1, 0.7, 0.45, 15, TEAL, True)
    flow(s, ["Document", "Chunk", "Embedding", "Chroma"], y=2.0,
         fills=[WHITE, PALE_BLUE, PALE_BLUE, PALE_GOLD])
    add_text(s, "在线", 0.75, 4.25, 0.7, 0.45, 15, CORAL, True)
    flow(s, ["Question", "Query Vector", "Top-k Chunks", "Prompt + LLM", "Answer"], y=4.12,
         fills=[WHITE, PALE_BLUE, PALE_GOLD, PALE_CORAL, WHITE])

    # 28 vector space
    s = base_slide(prs, 28, "08 · RAG", "同维度，不代表同一个语义空间")
    node(s, "文档向量\nModel A", 1.0, 2.25, 2.5, 1.1, PALE_BLUE, TEAL)
    node(s, "问题向量\nModel A", 1.0, 4.0, 2.5, 1.1, PALE_BLUE, TEAL)
    node(s, "可比较", 5.25, 3.1, 2.2, 1.1, PALE_GOLD, GOLD)
    add_line(s, 3.5, 2.8, 5.2, 3.5); add_line(s, 3.5, 4.55, 5.2, 3.7)
    node(s, "Model B\n即使也是 384 维", 9.0, 3.1, 2.75, 1.1, PALE_CORAL, CORAL)
    add_text(s, "换 Embedding → 必须重建索引\n换生成 LLM → 通常无需重建", 7.8, 5.05, 4.9, 0.85, 18, NAVY, True, align=PP_ALIGN.CENTER)

    # 29 Chroma positioning
    s = base_slide(prs, 29, "08 · RAG", "ChromaDB 不是玩具；本课使用的是轻量模式")
    flow(s, ["本地嵌入式\n教学 / 原型", "Client / Server\n自托管", "Chroma Cloud\n托管服务"], y=2.45,
         fills=[PALE_BLUE, PALE_GOLD, PALE_CORAL])
    bullets(s, ["当前 chroma_db 是本地数据目录", "生产还需认证、多租户、备份、高可用和监控"], y=4.45, h=1.6, size=19)

    # 30 retrieval eval
    s = base_slide(prs, 30, "08 · RAG", "检索和生成必须分开评估")
    two_cards(s, "Retrieval", ["正确 chunk 是否进入 top-k", "Recall@k / MRR / nDCG", "阈值、过滤、reranker"],
              "Generation", ["是否忠于上下文", "答案完整与引用准确", "拒答与安全性"])

    # 31 choice
    s = base_slide(prs, 31, "Decision", "微调不是数据库，RAG 也不改变模型行为")
    for y, need, choice, c in [
        (2.0, "事实频繁更新、需要引用", "RAG", TEAL),
        (3.15, "格式、语气、任务行为稳定适配", "LoRA", CORAL),
        (4.3, "同时需要知识与行为", "RAG + LoRA", GOLD),
    ]:
        add_rect(s, 1.0, y, 7.6, 0.82, WHITE, c)
        add_text(s, need, 1.3, y + 0.1, 7.0, 0.56, 18, INK, True)
        add_rect(s, 9.25, y, 2.8, 0.82, c, c)
        add_text(s, choice, 9.25, y + 0.1, 2.8, 0.56, 18, NAVY if c == GOLD else WHITE, True, align=PP_ALIGN.CENTER)

    # 32 Production
    s = base_slide(prs, 32, "Engineering", "跑通 Demo，只是工程工作的起点")
    for i, (head, c) in enumerate([("数据质量", TEAL), ("独立评估", GOLD), ("可恢复产物", CORAL),
                                   ("服务与监控", GREEN), ("权限与安全", TEAL), ("版本与成本", GOLD)]):
        x = 0.85 + (i % 3) * 4.15; y = 2.05 + (i // 3) * 1.75
        add_rect(s, x, y, 3.65, 1.18, WHITE, c)
        add_text(s, head, x, y + 0.2, 3.65, 0.68, 20, c, True, align=PP_ALIGN.CENTER)
    add_text(s, "每次都要问：这个结果能证明什么，不能证明什么？", 1.25, 5.72, 10.8, 0.55, 20, NAVY, True, align=PP_ALIGN.CENTER)

    # 33 Close
    s = prs.slides.add_slide(prs.slide_layouts[6]); set_bg(s, NAVY)
    add_text(s, "真正的学习，不是“运行成功”", 0.85, 1.15, 11.7, 0.8, 31, WHITE, True, align=PP_ALIGN.CENTER)
    flow(s, ["提出假设", "控制变量", "记录指标", "解释结果", "说明限制"], y=3.0,
         fills=[PALE_BLUE, PALE_GOLD, PALE_CORAL, PALE_BLUE, PALE_GOLD])
    add_text(s, "下一步：完成一个可评估、可恢复、边界清晰的结课项目", 1.15, 5.05, 11.0, 0.72, 21, "B8CEDD", True, align=PP_ALIGN.CENTER)
    add_text(s, "AI TRAINING LAB", 5.0, 6.52, 3.35, 0.4, 12, TEAL, True, font="Avenir Next", align=PP_ALIGN.CENTER)

    prs.save(OUTPUT)
    print(f"Generated {len(prs.slides)} slides: {OUTPUT}")


if __name__ == "__main__":
    build()
