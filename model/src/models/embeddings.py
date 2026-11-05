"""Extraction d'embeddings avec un CNN pré-entraîné (transfer learning)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

from ..dataset import IMG_SIZE, get_transforms


def build_backbone(name: str = "mobilenet_v3_small", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Retourne (backbone sans tête de classification, dimension des features)."""
    weights = "DEFAULT" if pretrained else None
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights)
        dim = m.classifier[3].in_features
        m.classifier = nn.Identity()
    elif name == "resnet18":
        m = models.resnet18(weights=weights)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"Modèle inconnu : {name}")
    m.eval()
    return m, dim


@torch.no_grad()
def extract_embeddings(
    loader,
    backbone: nn.Module,
    device: str = "cpu",
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Passe le DataLoader dans le backbone → (embeddings [N, D], labels [N])."""
    backbone = backbone.to(device)
    feats, labels = [], []
    n = 0
    for x, y in loader:
        x = x.to(device)
        f = backbone(x)
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
        n += len(y)
        if max_samples is not None and n >= max_samples:
            break
    return np.concatenate(feats)[:max_samples], np.concatenate(labels)[:max_samples]


def eval_transform():
    return get_transforms(train=False)


def load_image_tensor(path: Path | str, device: str = "cpu") -> torch.Tensor:
    from PIL import Image

    tf = eval_transform()
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)
