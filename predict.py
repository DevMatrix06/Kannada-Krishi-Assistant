#!/usr/bin/env python
"""
Quick CLI test for the disease classifier.

Usage:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --top 3
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

_WEIGHTS  = Path("models/disease_model.pt")
_CLASSES  = Path("models/disease_classes.json")
_MEAN     = [0.485, 0.456, 0.406]
_STD      = [0.229, 0.224, 0.225]

_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def load_model():
    for p in (_WEIGHTS, _CLASSES):
        if not p.exists():
            sys.exit(f"[ERROR] Missing {p} — run `python module1_disease/train.py` first.")

    with open(_CLASSES, encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(
        net.classifier[1].in_features, len(idx_to_class)
    )
    net.load_state_dict(torch.load(_WEIGHTS, map_location="cpu", weights_only=True))
    net.eval()
    return net, idx_to_class


def main():
    parser = argparse.ArgumentParser(description="Crop disease predictor")
    parser.add_argument("image", help="Path to a leaf photo (JPEG/PNG)")
    parser.add_argument("--top", type=int, default=1, metavar="N",
                        help="Show top-N predictions (default: 1)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"[ERROR] Image not found: {img_path}")

    print(f"Loading model...", end=" ", flush=True)
    model, idx_to_class = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"OK  ({device})")

    img    = Image.open(img_path).convert("RGB")
    tensor = _tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze()

    top_k = min(args.top, len(idx_to_class))
    values, indices = probs.topk(top_k)

    print(f"\nImage : {img_path.name}")
    print("-" * 40)
    for rank, (conf, idx) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        label = idx_to_class[idx]
        bar   = "█" * int(conf * 30)
        print(f"  {rank}. {label:<35}  {conf:6.2%}  {bar}")


if __name__ == "__main__":
    main()
