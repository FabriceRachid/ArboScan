"""Baseline : embeddings MobileNetV3 + k-NN (cosine) — rapide, idéal few-shot.

Usage :
    python -m model.src.models.baseline_knn
    python -m model.src.models.baseline_knn --k 7 --backbone mobilenet_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

from ..dataset import (
    METADATA_DIR,
    ROOT,
    SAMPLES_DIR,
    TreeLeafFruitDataset,
    build_labels,
    load_metadata,
    load_splits,
    _pick,
)
from .embeddings import build_backbone, extract_embeddings

MODELS_DIR = ROOT / "model" / "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_knn(
    backbone_name: str = "mobilenet_v3_small",
    k: int = 5,
    batch_size: int = 32,
) -> tuple[KNeighborsClassifier, dict]:
    splits = load_splits()
    labels = build_labels(splits)
    idx_to_slug = {i: s for s, i in labels.items()}

    backbone, dim = build_backbone(backbone_name, pretrained=True)
    train_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "train"), labels, train=False)
    val_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "val"), labels, train=False)
    test_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "test"), labels, train=False)

    print(f"Embeddings ({backbone_name}, dim={dim}) sur device={DEVICE}")
    print(f"  train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    X_train, y_train = extract_embeddings(train_loader, backbone, DEVICE)
    X_val, y_val = extract_embeddings(val_loader, backbone, DEVICE)
    X_test, y_test = extract_embeddings(test_loader, backbone, DEVICE)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(X_train, y_train)

    def acc(X, y):
        if len(y) == 0:
            return float("nan")
        return float((knn.predict(X) == y).mean())

    def topk(X, y, ktop=3):
        if len(y) == 0:
            return float("nan")
        proba = knn.predict_proba(X)
        kk = min(ktop, proba.shape[1])
        top = np.argsort(-proba, axis=1)[:, :kk]
        return float(np.mean([y[i] in top[i] for i in range(len(y))]))

    metrics = {
        "backbone": backbone_name,
        "k": k,
        "dim": dim,
        "train_acc": acc(X_train, y_train),
        "val_acc": acc(X_val, y_val),
        "test_top1": acc(X_test, y_test),
        "test_top3": topk(X_test, y_test, 3),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    print(
        f"kNN(k={k}) | val top1={metrics['val_acc']:.3f} | "
        f"test top1={metrics['test_top1']:.3f} top3={metrics['test_top3']:.3f}"
    )

    MODELS_DIR.mkdir(exist_ok=True)
    torch.save(backbone.state_dict(), MODELS_DIR / "baseline_knn_backbone.pt")
    joblib.dump(
        {
            "knn": knn,
            "backbone_name": backbone_name,
            "labels": labels,
            "idx_to_slug": idx_to_slug,
            "metrics": metrics,
        },
        MODELS_DIR / "baseline_knn.joblib",
    )
    (MODELS_DIR / "baseline_knn_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modèle sauvegardé → {MODELS_DIR / 'baseline_knn.joblib'}")
    return knn, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "mobilenet_v2", "resnet18", "efficientnet_b0"])
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    train_knn(a.backbone, a.k, a.batch_size)
