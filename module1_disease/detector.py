"""
Inference-only module for crop disease classification.
Weights must exist at models/disease_model.pt (run train.py first).
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

_WEIGHTS_PATH  = Path("models/disease_model.pt")
_CLASSES_PATH  = Path("models/disease_classes.json")

_IMG_SIZE = 224
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_infer_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# Lazy-loaded globals — initialised on first call to predict_disease()
_model: torch.nn.Module | None = None
_idx_to_class: dict[int, str] | None = None
_device: torch.device | None = None


def _load() -> None:
    global _model, _idx_to_class, _device

    if not _WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Weights not found at '{_WEIGHTS_PATH}'. Run module1_disease/train.py first."
        )
    if not _CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"Class map not found at '{_CLASSES_PATH}'. Run module1_disease/train.py first."
        )

    with open(_CLASSES_PATH, encoding="utf-8") as f:
        class_to_idx: dict[str, int] = json.load(f)
    _idx_to_class = {v: k for k, v in class_to_idx.items()}

    num_classes = len(_idx_to_class)
    net = models.efficientnet_b0(weights=None)
    net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_classes)
    net.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True))
    net.eval()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = net.to(_device)


def predict_disease(image_path: str) -> dict:
    """
    Classify a crop leaf photo.

    Args:
        image_path: Path to a JPEG or PNG leaf photo.

    Returns:
        {"label": str, "confidence": float}
        label is the PlantVillage class name, e.g. "Tomato___Late_blight".
        confidence is a softmax probability in [0, 1].

    Raises:
        FileNotFoundError: if the image or model weights are missing.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if _model is None:
        _load()

    img    = Image.open(path).convert("RGB")
    tensor = _infer_tf(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    return {
        "label":      _idx_to_class[idx.item()],
        "confidence": round(conf.item(), 4),
    }
