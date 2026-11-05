"""API ArboScan : identification d'arbre par photo (feuille ou fruit).

Lancement :
    uvicorn backend.app.main:app --reload --port 8000

Endpoints :
    GET  /health     → statut
    GET  /especes    → base de connaissances (noms + vertus)
    POST /predict    → image multipart → espèce + noms locaux + vertus
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model.src.dataset import ROOT, load_metadata
from model.src.inference import DEFAULT_CKPT, DEFAULT_KNN, Predictor, _fiche_from_row

app = FastAPI(title="ArboScan API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        mode = "cnn" if DEFAULT_CKPT.exists() else "knn"
        if mode == "knn" and not DEFAULT_KNN.exists():
            raise HTTPException(
                status_code=503,
                detail="Aucun modèle entraîné. Lance : python -m model.src.models.baseline_knn puis python -m model.src.train",
            )
        _predictor = Predictor(DEFAULT_CKPT, DEFAULT_KNN, mode=mode)
    return _predictor


@app.get("/health")
def health():
    p = None
    try:
        p = get_predictor()
        mode = p.mode
    except HTTPException:
        mode = None
    return {"status": "ok", "modele": mode}


@app.get("/especes")
def especes():
    df = load_metadata()
    if df.empty:
        raise HTTPException(status_code=404, detail="espèces.csv introuvable")
    return [_fiche_from_row(row) for _, row in df.iterrows()]


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = 3):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier attendu : image (jpg/png/webp)")
    try:
        predictor = get_predictor()
    except HTTPException as e:
        raise e
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        return predictor.predict(tmp_path, topk=max(1, min(topk, 10)))
    finally:
        tmp_path.unlink(missing_ok=True)
