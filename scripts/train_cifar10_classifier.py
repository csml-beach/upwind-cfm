#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lcfm.cifar_metrics import build_cifar_resnet18
from lcfm.utils import set_seed, write_json


def make_loader(root, train, batch_size, workers):
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda x: x),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    dataset = CIFAR10(root=root, train=train, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss_sum += float(loss.item()) * labels.numel()
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.numel())
    return {"loss": loss_sum / total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser(description="Train a small ResNet-18 CIFAR-10 classifier for conditional sample evaluation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/classifiers/cifar10_resnet18.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader = make_loader(args.data_root, True, args.batch_size, args.workers)
    test_loader = make_loader(args.data_root, False, args.batch_size, args.workers)
    model = build_cifar_resnet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best = {"accuracy": 0.0, "epoch": 0}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * labels.numel()
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += int(labels.numel())
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            **{f"test_{key}": value for key, value in evaluate(model, test_loader, device).items()},
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(row, flush=True)
        if row["test_accuracy"] > best["accuracy"]:
            best = {"accuracy": row["test_accuracy"], "epoch": epoch}
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": 10,
                    "architecture": "cifar_resnet18",
                    "best": best,
                    "history": history,
                    "normalization": {
                        "mean": [0.4914, 0.4822, 0.4465],
                        "std": [0.2470, 0.2435, 0.2616],
                    },
                },
                out_path,
            )

    write_json(out_path.with_suffix(".json"), {"best": best, "history": history, "checkpoint": str(out_path)})
    print(f"Saved classifier to {out_path} with best={best}")


if __name__ == "__main__":
    main()
