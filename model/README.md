# Modèle et données

- data/images/<slug>/ : photos regroupées par espèce.
- data/metadata/species.csv : noms, langues locales et informations d'usage.
- data/metadata/splits.json : partition train/validation/test.
- src/ : préparation, curation, apprentissage, évaluation et inférence.
- notebooks/ArboScan_entrainement.ipynb : notebook guidé.
- artifacts/ : checkpoints et métriques générés.

Les scripts se lancent depuis la racine avec le module model.src.*.
