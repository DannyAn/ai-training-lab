"""使用预训练 BERT 进行文本分类微调

使用 HuggingFace transformers 加载预训练 BERT，
在 GLUE SST-2 (情感二分类) 上微调。
"""

import torch
from torch import nn
from transformers import BertModel, BertTokenizer


class BertClassifier(nn.Module):
    """BERT + 分类头"""

    def __init__(self, model_name: str = "google-bert/bert-base-uncased", num_classes: int = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] token
        return self.classifier(self.dropout(pooled))


def load_tokenizer(model_name: str = "google-bert/bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    model = BertClassifier()
    dummy_ids = torch.randint(0, 100, (2, 128))
    dummy_mask = torch.ones(2, 128)
    out = model(dummy_ids, dummy_mask)
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  BERT frozen: 0,  Trainable classifier: {sum(p.numel() for p in model.classifier.parameters()):,}")
