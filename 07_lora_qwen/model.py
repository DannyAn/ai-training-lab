"""使用 LoRA / QLoRA 微调 Qwen 大语言模型

这份代码是「大模型微调」的最小化教学版：
1) 先加载一个预训练基座（Qwen）
2) 冻结原始权重，只注入一个很小的 LoRA adapter
3) 训练时只更新这部分适配器参数，从而达到“低成本改写模型行为”的目的

注意：完整运行需要 GPU (24GB+ 显存)，如果资源有限，可设置 use_4bit=True 使用 QLoRA。
Qwen 这种大语言模型的微调通常只训练 adapter，而不是整模型全部权重。
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


def create_lora_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    use_4bit: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
):
    """加载模型并应用 LoRA。

    这里是整个 LoRA 训练流程的入口函数：
    - 先把预训练 Qwen 大模型下载/加载进来
    - 如果 `use_4bit=True`，开启 QLoRA（4-bit 量化）来节省显存
    - 接着按 `LoraConfig` 规则，给关键线性层注入小适配器
    - 最后返回可继续训练的模型和 tokenizer
    """

    # 【记忆】QLoRA 的意义：
    #   不直接把整模型放到很高精度，先用 4 bit 量化压缩基座权重，
    #   这样在同样显存下可以训练更大的模型。
    bnb_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        )

    # 【记忆】`AutoModelForCausalLM.from_pretrained`：
    #   这是“从 HuggingFace 下载/加载一个自回归语言模型”的标准入口。
    #   `device_map="auto"` 会自动把模型分配到可用设备（GPU / CPU）。
    #   `quantization_config` 用来控制是否采用 4bit 量化。
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # 【记忆】tokenizer 是把文本转成模型能读的 token 序列的工具。
    #   `pad_token` 需要和 `eos_token` 对齐，否则生成/训练时可能出现 padding 问题。
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 【记忆】LoRA 最核心的配置：
    #   - `r`：LoRA 矩阵秩，越小越轻、越大越强，但也越容易过拟合。
    #   - `lora_alpha`：缩放因子，常见做法是 `alpha ≈ 2 * r`。
    #   - `target_modules`：告诉模型“把哪些投影层做 LoRA 改造”。
    #   这里直接作用在 Qwen 的 q/k/v/o 以及 MLP 的 gate/up/down 投影层。
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # 【记忆】`get_peft_model` 会把 LoRA adapter 注入到原模型中，
    #   并把所谓“可训练参数”只限制在这些 adapter 上，而不是整模型。
    model = get_peft_model(model, lora_config)

    # 【记忆】这是一个“训练参数扫描”输出：
    #   你会看到可训练参数数目通常远小于整模型参数数，
    #   这正是 LoRA 高效微调的本质。
    model.print_trainable_parameters()
    return model, tokenizer


if __name__ == "__main__":
    # 【使用】可以先单独跑这个 smoke-test，确认模型加载与 LoRA 注入成功。
    #   如果这里能顺利加载，后续训练环节的代码逻辑基本是对的。
    model, tokenizer = create_lora_model(use_4bit=False)
    print(f"\nModel type: {type(model).__name__}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
