"""Évaluation du modèle fine-tuné sur le split test : top-1, top-3, matrice de confusion.

Usage :
    python -m model.src.evaluate --ckpt model/artifacts/checkpoints/mobilenet_v3_small_best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from .dataset import ROOT, SAMPLES_DIR, TreeLeafFruitDataset, get_transforms, load_splits, _pick
from .train import build_classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "model" / "artifacts"


def evaluate(ckpt_path: Path, batch_size: int = 32) -> dict:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    backbone_name = ckpt["backbone_name"]
    labels = ckpt["labels"]
    idx_to_slug = {int(i): s for s, i in labels.items()}

    splits = load_splits()
    test_ds = TreeLeafFruitDataset(SAMPLES_DIR, _pick(splits, "test"), labels, train=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Test sur {len(test_ds)} images, {len(labels)} classes, device={DEVICE}")

    model = build_classifier(backbone_name, ckpt["num_classes"], pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in test_loader:
            all_logits.append(model(x.to(DEVICE)).cpu())
            all_y.append(y)
    logits = torch.cat(all_logits)
    y = torch.cat(all_y).numpy()
    prob = torch.softmax(logits, dim=1).numpy()

    top1_pred = prob.argmax(axis=1)
    top3_pred = np.argsort(-prob, axis=1)[:, :3]

    top1 = float((top1_pred == y).mean())
    top3 = float(np.mean([y[i] in top3_pred[i] for i in range(len(y))]))
    report = classification_report(
        [idx_to_slug[i] for i in y],
        [idx_to_slug[i] for i in top1_pred],
        zero_division=0,
    )
    cm = confusion_matrix(y, top1_pred, labels=np.arange(len(labels)))
    class_names = [idx_to_slug[i] for i in range(len(labels))]

    metrics = {
        "ckpt": str(ckpt_path),
        "backbone": backbone_name,
        "n_test": int(len(y)),
        "test_top1": top1,
        "test_top3": top3,
        "best_val_top1": ckpt.get("best_val_top1"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (OUT_DIR / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vrai_prédit", *class_names])
        for slug, row in zip(class_names, cm):
            writer.writerow([slug, *row.tolist()])

    row_totals = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    with (OUT_DIR / "confusion_matrix_normalized.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vrai_prédit", *class_names])
        for slug, row in zip(class_names, normalized):
            writer.writerow([slug, *(f"{value:.4f}" for value in row)])
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print(f"top1={top1:.3f} top3={top3:.3f}")
    print(report)
    faibles = [
        idx_to_slug[i] for i in range(len(labels)) if (cm[i].sum() > 0 and cm[i, i] / cm[i].sum() < 0.5)
    ]
    if faibles:
        print("[!] Classes avec rappel < 50% :", ", ".join(faibles))
    print(f"Artefacts → {OUT_DIR}/eval_metrics.json, confusion_matrix.csv, confusion_matrix_normalized.csv, classification_report.txt")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "model" / "artifacts" / "checkpoints" / "mobilenet_v3_small_best.pt"))
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    evaluate(Path(a.ckpt), a.batch_size)
