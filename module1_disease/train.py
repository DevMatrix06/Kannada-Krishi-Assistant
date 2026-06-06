"""
PlantVillage crop-disease classifier — EfficientNet-B0, transfer learning.

Dataset layout (pre-split, from archive.zip):
    data/plantvillage/train/<ClassName>/*.jpg
    data/plantvillage/valid/<ClassName>/*.jpg

GPU note: tested on RTX 3050 4 GB laptop.  BATCH_SIZE=32 fits comfortably
with AMP.  If you hit OOM, drop to 16.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ── hyper-parameters ─────────────────────────────────────────────────────────
TRAIN_DIR  = Path("data/plantvillage/train")
VALID_DIR  = Path("data/plantvillage/valid")
MODEL_OUT  = Path("models/disease_model.pt")
CLASSES_OUT = Path("models/disease_classes.json")

BATCH_SIZE   = 32   # lower to 16 if RTX 3050 OOMs
IMG_SIZE     = 224
LR           = 3e-4
WEIGHT_DECAY = 1e-2
EPOCHS       = 15
NUM_WORKERS  = 2    # safe on Windows because all DataLoader code is inside main()
PIN_MEMORY   = True

# ImageNet stats (EfficientNet was pretrained on ImageNet)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def build_loaders():
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds   = datasets.ImageFolder(VALID_DIR, transform=val_tf)

    # Both splits must have the same classes in the same order.
    if train_ds.classes != val_ds.classes:
        only_train = set(train_ds.classes) - set(val_ds.classes)
        only_valid = set(val_ds.classes) - set(train_ds.classes)
        sys.exit(
            f"[ERROR] train/valid class mismatch.\n"
            f"  Only in train: {only_train}\n  Only in valid: {only_valid}"
        )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    return train_loader, val_loader, train_ds.class_to_idx


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    # Replace head for our class count
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast("cuda"):
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


def main():
    for split_dir in (TRAIN_DIR, VALID_DIR):
        if not split_dir.exists():
            sys.exit(
                f"[ERROR] Dataset split not found at '{split_dir}'.\n"
                "Expected layout: data/plantvillage/train/<Class>/ and data/plantvillage/valid/<Class>/"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_loader, val_loader, class_to_idx = build_loaders()
    num_classes = len(class_to_idx)
    print(f"Classes: {num_classes}  |  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    with open(CLASSES_OUT, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    print(f"Class map saved -> {CLASSES_OUT}")

    model     = build_model(num_classes, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler("cuda")

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        acc  = evaluate(model, val_loader, device)
        scheduler.step()
        marker = ""
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
            marker = "  * saved"
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  val_acc={acc:.4f}{marker}", flush=True)

    print(f"\nDone. Best val_acc: {best_acc:.4f}  Weights -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
