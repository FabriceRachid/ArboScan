"""
Script v2 - Vérification approfondie + téléchargement d'échantillon GBIF.

Améliorations par rapport à la v1 :
  - Filtre basisOfRecord=HUMAN_OBSERVATION (exclut les planches d'herbier)
  - Récupère et résume les licences (CC0, CC-BY, CC-BY-NC, etc.)
  - Télécharge un échantillon de N photos par espèce dans un dossier
    pour que tu puisses les inspecter visuellement toi-même
    (est-ce vraiment une feuille/fruit en gros plan et net ?)

Usage :
    pip install requests
    python model/scripts/check_dataset_v2.py model/data/metadata/species.csv --sample 20

Le CSV doit avoir une colonne 'nom_scientifique'.

Sorties :
  - especes_disponibilite_v2.csv : statistiques détaillées par espèce
  - model/data/images/<nom_espece>/*.jpg : échantillon de photos téléchargées
"""

import argparse
import csv
import os
import re
import time
import requests

PAYS_AFRIQUE_OUEST = ["BF", "ML", "NE", "SN", "CI", "GH", "TG", "BJ", "NG", "GN"]

GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_SPECIES_MATCH_URL = "https://api.gbif.org/v1/species/match"


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip()).strip("_").lower()


def get_taxon_key(nom_scientifique: str):
    try:
        resp = requests.get(GBIF_SPECIES_MATCH_URL, params={"name": nom_scientifique}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("usageKey")
    except requests.RequestException as e:
        print(f"  [erreur match taxon] {nom_scientifique}: {e}")
        return None


def fetch_occurrences(taxon_key, limit=300, human_only=True):
    """Récupère jusqu'à `limit` occurrences avec photo, filtrées sur l'Afrique de l'Ouest."""
    all_records = []
    for pays in PAYS_AFRIQUE_OUEST:
        offset = 0
        page_size = 100
        while True:
            params = {
                "taxonKey": taxon_key,
                "mediaType": "StillImage",
                "country": pays,
                "limit": page_size,
                "offset": offset,
            }
            if human_only:
                params["basisOfRecord"] = "HUMAN_OBSERVATION"
            try:
                resp = requests.get(GBIF_OCCURRENCE_URL, params=params, timeout=20)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                print(f"  [erreur fetch {pays}] taxonKey={taxon_key}: {e}")
                break

            results = data.get("results", [])
            all_records.extend(results)
            offset += page_size
            if offset >= data.get("count", 0) or len(all_records) >= limit:
                break
            time.sleep(0.15)
        if len(all_records) >= limit:
            break
        time.sleep(0.15)
    return all_records[:limit]


def summarize_licenses(records):
    licenses = {}
    for r in records:
        lic = r.get("license", "inconnue")
        licenses[lic] = licenses.get(lic, 0) + 1
    return licenses


def download_sample(records, species_name, out_dir, sample_size=20):
    folder = os.path.join(out_dir, slugify(species_name))
    os.makedirs(folder, exist_ok=True)
    downloaded = 0
    for r in records:
        if downloaded >= sample_size:
            break
        media_list = r.get("media", [])
        for m in media_list:
            url = m.get("identifier")
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                ext = ".jpg"
                filename = f"{downloaded:03d}_{r.get('key')}{ext}"
                with open(os.path.join(folder, filename), "wb") as f:
                    f.write(resp.content)
                downloaded += 1
                break  # une photo par occurrence suffit pour l'échantillon
            except requests.RequestException:
                continue
        time.sleep(0.1)
    return downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="CSV avec colonne nom_scientifique")
    parser.add_argument("--sample", type=int, default=20, help="Nombre de photos à télécharger par espèce")
    parser.add_argument("--out", default="model/data/images", help="Dossier de sortie pour les échantillons")
    parser.add_argument("--no-download", action="store_true", help="Ne télécharge pas les photos, juste les stats")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        especes = [row["nom_scientifique"].strip() for row in reader if row.get("nom_scientifique")]

    print(f"{len(especes)} espèces à analyser en détail (filtrage HUMAN_OBSERVATION).\n")

    rapport = []
    for nom in especes:
        print(f"-> {nom}")
        taxon_key = get_taxon_key(nom)
        if taxon_key is None:
            rapport.append({
                "nom_scientifique": nom, "taxon_key": "", "nb_human_observation": 0,
                "licences": "", "photos_telechargees": 0, "statut_v2": "taxon non trouvé",
            })
            continue

        records = fetch_occurrences(taxon_key, limit=300, human_only=True)
        nb = len(records)
        licenses = summarize_licenses(records)
        licenses_str = "; ".join(f"{k}:{v}" for k, v in licenses.items())

        downloaded = 0
        if not args.no_download and nb > 0:
            downloaded = download_sample(records, nom, args.out, sample_size=args.sample)

        statut = "BON" if nb >= 150 else ("FAIBLE - à compléter" if nb >= 30 else "CRITIQUE - collecte terrain prioritaire")

        rapport.append({
            "nom_scientifique": nom,
            "taxon_key": taxon_key,
            "nb_human_observation": nb,
            "licences": licenses_str,
            "photos_telechargees": downloaded,
            "statut_v2": statut,
        })
        print(f"   -> {nb} occurrences HUMAN_OBSERVATION avec photo | {downloaded} téléchargées | licences: {licenses_str}")
        time.sleep(0.2)

    out_csv = "especes_disponibilite_v2.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["nom_scientifique", "taxon_key", "nb_human_observation", "licences", "photos_telechargees", "statut_v2"])
        writer.writeheader()
        writer.writerows(rapport)

    print(f"\nRapport détaillé écrit dans : {out_csv}")
    if not args.no_download:
        print(f"Échantillons photo téléchargés dans : {args.out}/<nom_espece>/")
        print("\n=> Inspecte visuellement ces dossiers : sont-ce de vraies photos de feuilles/fruits nettes,")
        print("   ou des photos d'herbier / d'arbre entier / floues ? Ça déterminera la vraie utilisabilité.")

    print("\nRésumé trié (du plus critique au plus solide) :")
    for r in sorted(rapport, key=lambda x: x["nb_human_observation"]):
        print(f"  {r['nom_scientifique']:30s} | {r['nb_human_observation']:4} obs. humaines | {r['statut_v2']}")


if __name__ == "__main__":
    main()
