# ArboScan

Application d'identification d'arbres à partir d'une photo de feuille ou de fruit. Le modèle prédit une espèce, puis l'API associe cette prédiction aux noms français/locaux et aux informations ethnobotaniques du catalogue.

## Structure

    frontend/                 Interface utilisateur (à construire)
    backend/
      app/main.py             API FastAPI
    model/
      data/
        images/<espèce>/      Images d'entraînement classées par dossier
        metadata/              CSV des espèces, splits et rapport de collecte
      notebooks/               Parcours Jupyter de préparation et entraînement
      src/
        dataset.py             Dataset, transformations et split stratifié
        train.py               Fine-tuning MobileNet
        evaluate.py            Évaluation sur le split test
        inference.py           Prédiction et fiche espèce
        curation/               Collecte d'images
        models/                 Baseline k-NN et embeddings
      artifacts/                Checkpoints, métriques et rapports
      scripts/                  Scripts de contrôle de dataset
    requirements.txt           Dépendances Python du projet

## Installation (PowerShell)

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

## Entraîner le modèle

Depuis la racine du dépôt :

    .\.venv\Scripts\python.exe -m model.src.dataset --prepare
    .\.venv\Scripts\python.exe -m model.src.train --backbone mobilenet_v3_small --epochs1 15 --epochs2 20
    .\.venv\Scripts\python.exe -m model.src.evaluate --ckpt model/artifacts/checkpoints/mobilenet_v3_small_best.pt

Le notebook model/notebooks/ArboScan_entrainement.ipynb présente ces étapes avec du code et des explications. Pour JupyterLab, installez requirements-notebook.txt puis lancez jupyter lab model/notebooks/ArboScan_entrainement.ipynb. Le notebook peut aussi être ouvert dans VS Code. Le code métier reste dans model/src/ pour être aussi utilisable depuis la ligne de commande et l'API.

## Lancer l'API

    .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000

Routes : GET /health, GET /especes, POST /predict.

## Données et modèles

Les images sont conservées localement sous model/data/images/. Les fichiers de poids .pt et le modèle k-NN .joblib sont sous model/artifacts/. Les gros fichiers de données et de modèles locaux sont exclus du suivi Git par .gitignore; les scripts, métadonnées et rapports restent lisibles dans le dépôt.
