"""Fine-tuning transfer learning (MobileNetV3 / MobileNetV2 / ResNet18 / EfficientNetB0).

Usage :
    python -m model.src.train
    python -m model.src.train --backbone mobilenet_v2 --epochs1 15 --epochs2 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import (
    ROOT,
    SAMPLES_DIR,
    TreeLeafFruitDataset,
    build_labels,
    get_transforms,
    load_splits,
    _pick,
)

MODELS_DIR = ROOT / "model" / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_classifier(backbone_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    from torchvision import models

    weights = "DEFAULT" if pretrained else None
    if backbone_name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=weights)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
    elif backbone_name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif backbone_name == "resnet18":
        m = models.resnet18(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif backbone_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(backbone_name)
    return m


def set_backbone_trainable(model: nn.Module, backbone_name: str, trainable: bool) -> None:
    """Gèle/dégèle le backbone en laissant la tête de classification entraînable."""
    if backbone_name in ("mobilenet_v3_small", "mobilenet_v2", "efficientnet_b0"):
        params = list(model.features.parameters())
    else:
        params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    for p in params:
        p.requires_grad = trainable


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, correct3, loss_sum = 0, 0, 0, 0.0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            top = logits.topk(min(3, logits.size(1)), dim=1).indices
            correct += (top[:, 0] == y).sum().item()
            correct3 += (top == y.unsqueeze(1)).any(dim=1).sum().item()
            total += len(y)
    return {
        "loss": loss_sum / max(total, 1),
        "top1": correct / max(total, 1),
        "top3": correct3 / max(total, 1),
        "n": total,
    }


def train(
    backbone_name: str = "mobilenet_v3_small",
    epochs1: int = 15,
    epochs2: int = 20,
    batch_size: int = 32,
    lr1: float = 1e-3,
    lr2: float = 1e-4,
    weight_decay: float = 1e-4,
) -> Path:
    splits = load_splits()
    labels = build_labels(splits)
    idx_to_slug = {i: s for s, i in labels.items()}
    num_classes = len(labels)

    train_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "train"), labels, train=True)
    val_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "val"), labels, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Classes={num_classes} train={len(train_ds)} val={len(val_ds)} device={DEVICE}")

    model = build_classifier(backbone_name, num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, best_state, best_epoch = 0.0, None, -1
    history = []

    # --- Phase 1 : seule la tête est entraînée ---
    print(f"\nPhase 1 — tête seule ({epochs1} époques, lr={lr1})")
    set_backbone_trainable(model, backbone_name, trainable=False)
    opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr1, weight_decay=weight_decay)
    for epoch in range(epochs1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, criterion, DEVICE, opt)
        va = _run_epoch(model, val_loader, criterion, DEVICE, None)
        history.append({"phase": 1, "epoch": epoch, "train": tr, "val": va})
        mark = " *" if va["top1"] > best_val else ""
        if va["top1"] > best_val:
            best_val, best_epoch = va["top1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  ep{epoch:02d} loss={tr['loss']:.3f} train_top1={tr['top1']:.3f} "
            f"val_top1={va['top1']:.3f} val_top3={va['top3']:.3f} ({time.time()-t0:.1f}s){mark}"
        )

    # --- Phase 2 : dégel partiel du backbone (si epochs2 > 0) ---
    if epochs2 > 0:
        print(f"\nPhase 2 — fine-tuning partiel ({epochs2} époques, lr={lr2})")
        if best_state:
            model.load_state_dict(best_state)
        set_backbone_trainable(model, backbone_name, trainable=True)
        opt = optim.AdamW(model.parameters(), lr=lr2, weight_decay=weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs2)
        for epoch in range(epochs2):
            t0 = time.time()
            tr = _run_epoch(model, train_loader, criterion, DEVICE, opt)
            va = _run_epoch(model, val_loader, criterion, DEVICE, None)
            sched.step()
            history.append({"phase": 2, "epoch": epoch, "train": tr, "val": va})
            mark = " *" if va["top1"] > best_val else ""
            if va["top1"] > best_val:
                best_val, best_epoch = va["top1"], epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"  ep{epoch:02d} loss={tr['loss']:.3f} train_top1={tr['top1']:.3f} "
                f"val_top1={va['top1']:.3f} val_top3={va['top3']:.3f} ({time.time()-t0:.1f}s){mark}"
            )

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt = MODELS_DIR / "checkpoints" / f"{backbone_name}_best.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "backbone_name": backbone_name,
            "num_classes": num_classes,
            "labels": labels,
            "idx_to_slug": idx_to_slug,
            "best_val_top1": best_val,
            "best_epoch": best_epoch,
            "img_size": 224,
        },
        ckpt,
    )
    (MODELS_DIR / "checkpoints" / f"{backbone_name}_history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\nMeilleur val top1 = {best_val:.3f} (époque {best_epoch})")
    print(f"Checkpoint → {ckpt}")
    return ckpt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "mobilenet_v2", "resnet18", "efficientnet_b0"])
    p.add_argument("--epochs1", type=int, default=15)
    p.add_argument("--epochs2", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr1", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=1e-4)
    a = p.parse_args()
    train(a.backbone, a.epochs1, a.epochs2, a.batch_size, a.lr1, a.lr2)
