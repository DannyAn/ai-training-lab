"""通用训练循环"""

import time
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """通用训练器，支持分类和生成任务"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        num_epochs: int = 10,
        callbacks: Optional[list[Callable]] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.num_epochs = num_epochs
        self.callbacks = callbacks or []
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    @torch.inference_mode()
    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y in tqdm(self.val_loader, desc="Val", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        return total_loss / len(self.val_loader), correct / total

    def fit(self):
        best_acc = 0.0
        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            elapsed = time.time() - t0
            print(f"Epoch {epoch}/{self.num_epochs}  |  train_loss={train_loss:.4f}  |  {elapsed:.1f}s")

            self.history["train_loss"].append(train_loss)

            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate()
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                arrow = " ↑" if val_acc > best_acc else ""
                print(f"              val_loss={val_loss:.4f}  val_acc={val_acc:.2%}{arrow}")
                if val_acc > best_acc:
                    best_acc = val_acc

            for cb in self.callbacks:
                cb(self)

        return self.history
