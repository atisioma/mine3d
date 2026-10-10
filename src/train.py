"""
训练脚本 — PointNet 语义分割 (室内 3 类)
用法: python -m src.train [--epochs 50] [--batch-size 4] [--lr 1e-3]
"""
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm

from src.model import build_model
from src.data import build_dataloader


CLASSES = {0: "table", 1: "apple", 2: "banana"}
NUM_CLASSES = 3


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)   # (B, N, 6)
        labels = batch["labels"].to(device)   # (B, N)
        masks = batch["masks"].to(device)     # (B, N)

        logits = model(inputs)                # (B, N, C)

        # 只计算有效点的 loss
        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss = (loss * masks.reshape(-1).float()).sum() / masks.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * masks.sum().item()

        preds = logits.argmax(dim=-1)
        correct += ((preds == labels) * masks).sum().item()
        total += masks.sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        masks = batch["masks"].to(device)

        logits = model(inputs)

        loss = criterion(logits.reshape(-1, NUM_CLASSES), labels.reshape(-1))
        loss = (loss * masks.reshape(-1).float()).sum() / masks.sum()

        total_loss += loss.item() * masks.sum().item()
        preds = logits.argmax(dim=-1)
        correct += ((preds == labels) * masks).sum().item()
        total += masks.sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 数据
    train_loader = build_dataloader(args.data_dir, "train", args.batch_size, args.num_workers)
    val_loader = build_dataloader(args.data_dir, "val", args.batch_size, args.num_workers)
    print(f"Train: {len(train_loader.dataset)} samples, Val: {len(val_loader.dataset)} samples")

    # 模型
    model = build_model(num_classes=NUM_CLASSES, in_ch=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # 损失 / 优化器 / 调度器
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # 保存 best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best.pth")
            print(f"  -> saved best (acc={best_acc:.4f})")

    print(f"\nTraining done. Best val acc: {best_acc:.4f}")
    torch.save(model.state_dict(), save_dir / "last.pth")


if __name__ == "__main__":
    main()
