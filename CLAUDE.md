# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A hands-on deep-learning learning repo: 8 progressive modules (`01_mnist` … `08_rag_demo`) built from scratch, each independently runnable. Code is illustrative/educational, not production. Comments and docstrings are in Chinese.

## Commands

Run a module (always from repo root — see Architecture for why):

```bash
python 01_mnist/train.py        # any module: python <module>/train.py
python 01_mnist/model.py        # smoke-test model forward pass / shape
```

Environment setup — there is a local `ai-env/` venv (Python 3.12, created from anaconda3):

```bash
source ai-env/bin/activate
pip install -r requirements.txt
```

Two requirements files exist and diverge: root `requirements.txt` (canonical, newer — adds safetensors, jupyterlab, pandas, scikit-learn) and `common/requirements.txt` (referenced by README, adds RAG stack: langchain/chromadb/sentence-transformers/einops). Install both if working across all modules.

There is no test suite and no linter. Verification = running the relevant `train.py` end-to-end and checking the printed loss/accuracy.

## Architecture

### Module layout (uniform across all 8 modules)
Each module dir contains:
- `model.py` — model definition(s); runnable standalone for a shape check.
- `train.py` — entry point; constructs model + data + optimizer and runs training.
- `__init__.py` — empty.

### The run-from-root convention
Every `train.py` starts with `sys.path.append(".")`. This makes `from common.utils import ...` and `from model import ...` resolve **only when invoked from the repo root**. Data paths are also relative (`./data`, `./<module>/output`), so running from inside a module dir breaks both imports and data loading.

### `common/` — shared utilities
- `utils.py` — `set_seed(42)` for reproducibility; `get_device()` auto-selects **CUDA → MPS (Apple Silicon) → CPU**; `count_parameters`.
- `trainer.py` — generic `Trainer` class (`fit` / `train_epoch` / `evaluate`), tracks a `history` dict (`train_loss`/`val_loss`/`val_acc`).
- `data.py` — `get_mnist_loaders`, `get_cifar10_loaders` (with augmentation).

### When `common.Trainer` applies — and when it doesn't
`common.Trainer` is **classification-shaped**: hard-coded `CrossEntropyLoss` + `argmax` accuracy, `(x, y)` batch unpacking. Used by `01_mnist`, `02_cifar10`. Anything else bypasses it:
- `04_transformer_from_scratch`, `05_gpt_tiny` — custom training loops (seq2seq / autoregressive).
- `06_bert_finetune`, `07_lora_qwen` — HuggingFace `Trainer` / `TrainingArguments`.
- `08_rag_demo` — no training loop; retrieval demo.

When adding a classification module, reach for `common.Trainer`; otherwise write a custom loop rather than forcing it.

### Device & backend notes
- Apple Silicon is a first-class target (`mps` path in `get_device()`).
- `bitsandbytes` (QLoRA 4-bit) is **Linux CUDA only** — it will not import on macOS. `07_lora_qwen/train.py` defaults to `use_4bit=False` for this reason; full QLoRA needs a CUDA GPU with 12–24GB+.
- `04_transformer_from_scratch` and `07_lora_qwen` use **toy/dummy datasets** for demonstration; the docstrings note that real training needs multi-GPU.

### Data & artifacts (all gitignored)
- `data/` — auto-downloaded datasets (MNIST, CIFAR-10). `data/` is in `.gitignore`.
- `ai-env/` — local venv, gitignored.
- Per-module `output/` dirs — checkpoints / LoRA weights / ChromaDB stores, written at runtime.
- `*.pt`, `*.pth`, `*.bin` — model checkpoints, gitignored.
