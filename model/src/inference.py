"""Inference : photo → espèce → fiche multilingue (français + noms locaux + vertus).

Usage :
    python -m model.src.inference model/data/images/adansonia_digitata/000_6185077192.jpg
    python -m model.src.inference photo.jpg --ckpt model/artifacts/checkpoints/mobilenet_v3_small_best.pt --topk 3
    python -m model.src.inference photo.jpg --mode knn
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image

from .dataset import CSV_PATH, ROOT, load_metadata
from .train import build_classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CKPT = ROOT / "model" / "artifacts" / "checkpoints" / "mobilenet_v3_small_best.pt"
DEFAULT_KNN = ROOT / "model" / "artifacts" / "baseline_knn.joblib"


def _fiche_from_row(row) -> dict:
    if row is None:
        return {}
    def g(col):
        v = row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        v = str(v).strip()
        return None if v in ("—", "-", "", "nan") else v
    return {
        "nom_scientifique": g("nom_scientifique"),
        "nom_francais": g("nom_francais"),
        "noms_locaux": {
            "moore": g("nom_mooré"),
            "dioula": g("nom_dioula"),
            "kassena": g("nom_kassena"),
            "san": g("nom_san"),
        },
        "famille": g("famille"),
        "parties_utilisees": g("parties_utilisees") or g("partie_comestible"),
        "vertus_medicinales": g("vertus_medicinales"),
        "vertus_alimentaires": g("vertus_alimentaires"),
        "usage_principal": g("usage_principal"),
    }


class Predictor:
    def __init__(self, ckpt_path: Path = DEFAULT_CKPT, knn_path: Path = DEFAULT_KNN, mode: str = "cnn"):
        self.mode = mode
        self.meta = load_metadata()
        self.meta_by_slug = {r["slug"]: r for _, r in self.meta.iterrows()} if len(self.meta) else {}

        if mode == "knn":
            if not knn_path.exists():
                raise FileNotFoundError(f"{knn_path} introuvable — lance python -m model.src.models.baseline_knn")
            blob = joblib.load(knn_path)
            self.knn = blob["knn"]
            self.labels = blob["labels"]
            self.idx_to_slug = blob["idx_to_slug"]
            from .models.embeddings import build_backbone

            self.backbone, _ = build_backbone(blob["backbone_name"], pretrained=False)
            backbone_pt = knn_path.parent / "baseline_knn_backbone.pt"
            if backbone_pt.exists():
                self.backbone.load_state_dict(torch.load(backbone_pt, map_location="cpu"))
            self.backbone.to(DEVICE).eval()
            self.model = None
        else:
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"{ckpt_path} introuvable — lance d'abord python -m model.src.train "
                    f"ou utilise --mode knn"
                )
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            self.labels = ckpt["labels"]
            self.idx_to_slug = {int(i): s for s, i in ckpt["labels"].items()}
            self.model = build_classifier(ckpt["backbone_name"], ckpt["num_classes"], pretrained=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(DEVICE).eval()
            self.backbone = None
            self.knn = None

    def _preprocess(self, image_path: Path) -> torch.Tensor:
        from .dataset import get_transforms

        tf = get_transforms(train=False)
        img = Image.open(image_path).convert("RGB")
        return tf(img).unsqueeze(0).to(DEVICE)

    def predict(self, image_path: str | Path, topk: int = 3) -> dict:
        image_path = Path(image_path)
        x = self._preprocess(image_path)
        with torch.no_grad():
            if self.mode == "knn":
                f = self.backbone(x).cpu().numpy()
                proba = self.knn.predict_proba(f)[0]
                classes = self.knn.classes_
            else:
                logits = self.model(x)
                proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
                classes = np.arange(len(proba))
        order = np.argsort(-proba)[:topk]
        predictions = []
        for rank, i in enumerate(order, 1):
            slug = self.idx_to_slug.get(int(classes[i]), str(classes[i]))
            conf = float(proba[i])
            fiche = _fiche_from_row(self.meta_by_slug.get(slug))
            predictions.append(
                {
                    "rang": rank,
                    "slug": slug,
                    "confiance": round(conf, 4),
                    "confiance_pct": f"{conf*100:.1f}%",
                    **fiche,
                }
            )
        best = predictions[0]
        noms = [v for v in (best.get("noms_locaux") or {}).values() if v]
        return {
            "image": str(image_path),
            "mode": self.mode,
            "confiance_seuil_alerte": best["confiance"] < 0.5,
            "message_alerte": (
                "Confiance faible — essayez une photo plus nette de la feuille ou du fruit seul."
                if best["confiance"] < 0.5
                else None
            ),
            "resultat_principal": {
                "espece": best.get("nom_scientifique") or best["slug"],
                "nom_francais": best.get("nom_francais"),
                "noms_locaux": best.get("noms_locaux"),
                "noms_locaux_reunis": " / ".join(noms) if noms else None,
                "famille": best.get("famille"),
                "confiance": best["confiance_pct"],
                "vertus_medicinales": best.get("vertus_medicinales"),
                "vertus_alimentaires": best.get("vertus_alimentaires"),
                "parties_utilisees": best.get("parties_utilisees"),
            },
            "top_k": predictions,
        }


def main():
    p = argparse.ArgumentParser(description="Identification d'arbre par feuille/fruit")
    p.add_argument("image", help="Chemin vers la photo (feuille ou fruit)")
    p.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p.add_argument("--knn", default=str(DEFAULT_KNN))
    p.add_argument("--mode", choices=["cnn", "knn"], default="cnn")
    p.add_argument("--topk", type=int, default=3)
    a = p.parse_args()

    pred = Predictor(Path(a.ckpt), Path(a.knn), a.mode)
    result = pred.predict(a.image, topk=a.topk)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
