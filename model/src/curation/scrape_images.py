"""Scrape d'images libres (feuilles/fruits) pour chaque espèce d'ArboScan.

Sources (licences ouvertes) :
  1. iNaturalist  — API pubrique, photos research-grade, licence CC-BY / CC-BY-SA / CC0
  2. Wikimedia Commons — images domaine public ou CC

Usage :
    python -m model.src.curation.scrape_images
    python -m model.src.curation.scrape_images --per-species 60 --only adansonia_digitata
    python -m model.src.curation.scrape_images --sources inat
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from ..dataset import CSV_PATH, ROOT, SAMPLES_DIR, discover_classes, slugify

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ArboScan/0.1 (recherche ethnobotanique Burkina Faso)"})

REPORT_PATH = ROOT / "model" / "data" / "metadata" / "scrape_report.json"
ATTRIBUTION_PATH = ROOT / "model" / "data" / "metadata" / "image_sources.jsonl"
INAT_LICENSES = "cc0,cc-by,cc-by-sa"
MIN_SIZE = 224  # côté min en pixels pour être utile


def _get_json(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"    [erreur] {e}")
                return None
            time.sleep(1 + attempt)
    return None


def _allowed_inat_license(code: str | None) -> bool:
    return (code or "").lower() in {"cc0", "cc-by", "cc-by-sa"}


def _allowed_commons_license(name: str, url: str) -> bool:
    text = f"{name} {url}".lower().replace("-", " ")
    if any(term in text for term in ("non commercial", "noncommercial", "nc", "nd", "fair use")):
        return False
    return (
        "cc0" in text
        or "public domain" in text
        or "creativecommons.org/licenses/by/" in text
        or "creativecommons.org/licenses/by sa/" in text
        or "cc by " in text
        or "cc by-sa " in text
    )


def _manual_exclusions(slug: str) -> int:
    if not ATTRIBUTION_PATH.exists():
        return 0
    count = 0
    for line in ATTRIBUTION_PATH.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += row.get("slug") == slug and row.get("review_status") == "excluded_manual"
    return count


def fetch_inaturalist(scientific_name: str, limit: int = 60) -> list[dict]:
    """Photos iNaturalist (observation → photo). Priorité aux feuilles/fruits/fleurs."""
    out = []
    seen_photo_ids = set()
    for quality in ("research", "needs_id", None):
        if len(out) >= limit:
            break
        params = {
            "taxon_name": scientific_name,
            "photos": "true",
            "license": INAT_LICENSES,
            "per_page": min(limit, 100),
            "order_by": "votes",
            "order": "desc",
        }
        if quality:
            params["quality_grade"] = quality
        data = _get_json("https://api.inaturalist.org/v1/observations", params)
        if not data or not data.get("results"):
            continue
        for obs in data["results"]:
            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                pid = photo.get("id")
                license_code = photo.get("license_code")
                if not url or pid in seen_photo_ids or not _allowed_inat_license(license_code):
                    continue
                seen_photo_ids.add(pid)
                url_large = re.sub(r"/square\.", "/large.", url)
                out.append(
                    {
                        "url": url_large,
                        "source": "inaturalist",
                        "license": license_code,
                        "attribution": photo.get("attribution"),
                        "photo_id": pid,
                        "page": f"https://www.inaturalist.org/observations/{obs.get('id')}"
                        if obs.get("id")
                        else "https://www.inaturalist.org/",
                    }
                )
                if len(out) >= limit:
                    return out
        time.sleep(0.3)
    return out


def fetch_wikimedia(scientific_name: str, limit: int = 40) -> list[dict]:
    """Wikimedia Commons : recherche images + images de pages d'espèce."""
    out = []
    queries = [
        f"{scientific_name} leaf",
        f"{scientific_name} fruit",
        scientific_name,
    ]
    seen_urls = set()
    for q in queries:
        if len(out) >= limit:
            break
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": f'filetype:bitmap {q}',
            "gsrnamespace": 6,  # namespace File
            "gsrlimit": 20,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": 1280,
            "format": "json",
        }
        data = _get_json("https://commons.wikimedia.org/w/api.php", params)
        if not data or "query" not in data:
            time.sleep(0.4)
            continue
        pages = data["query"].get("pages", {})
        for page in pages.values():
            ii = (page.get("imageinfo") or [{}])[0]
            thumb = ii.get("thumburl") or ii.get("url")
            if not thumb or thumb in seen_urls:
                continue
            w, h = ii.get("width", 0), ii.get("height", 0)
            if w < MIN_SIZE or h < MIN_SIZE:
                continue
            meta = ii.get("extmetadata", {})
            lic = (meta.get("LicenseShortName", {}) or {}).get("value", "")
            lic_url = (meta.get("LicenseUrl", {}) or {}).get("value", "")
            if not _allowed_commons_license(lic, lic_url):
                continue
            seen_urls.add(thumb)
            out.append(
                {
                    "url": thumb,
                    "source": "wikimedia",
                    "license": lic,
                    "license_url": lic_url,
                    "attribution": (meta.get("Artist", {}) or {}).get("value", ""),
                    "page": page.get("title"),
                    "page_url": "https://commons.wikimedia.org/wiki/"
                    + quote(page.get("title", "").replace(" ", "_"), safe="/:"),
                }
            )
            if len(out) >= limit:
                break
        time.sleep(0.4)
    return out


def download_one(url: str, dest_dir: Path, prefix: str, idx: int) -> Path | None:
    try:
        r = SESSION.get(url, timeout=12)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
        img.load()
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            return None
        # évite les doublons exacts (même contenu)
        digest = hashlib.md5(r.content).hexdigest()[:10]
        existing = {f.stem.split("__")[-1] for f in dest_dir.glob("*.jpg")}
        if digest in existing:
            return None
        if img.mode != "RGB":
            img = img.convert("RGB")
        # downscale si énorme (>2000px) pour économiser l'espace
        if max(w, h) > 2000:
            img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        dest = dest_dir / f"{prefix}_{idx:03d}__{digest}.jpg"
        img.save(dest, "JPEG", quality=90)
        return dest
    except Exception:
        return None


def scrape_species(
    scientific_name: str,
    per_species: int = 60,
    sources: tuple[str, ...] = ("inat", "wikimedia"),
) -> dict:
    slug = slugify(scientific_name)
    dest_dir = SAMPLES_DIR / slug
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest_dir.glob("*.jpg")))
    target = max(0, per_species - existing)

    candidates: list[dict] = []
    fetch_deadline = time.time() + 45
    if "inat" in sources:
        candidates += fetch_inaturalist(scientific_name, limit=target)
    if "wikimedia" in sources and len(candidates) < target and time.time() < fetch_deadline:
        candidates += fetch_wikimedia(scientific_name, limit=target - len(candidates) + 10)

    downloaded = 0
    skipped = 0
    deadline = time.time() + 90  # 90s max par espèce (anti-blocage réseau)
    for i, cand in enumerate(candidates):
        if downloaded >= target or time.time() > deadline:
            break
        prefix = cand["source"][:3]
        p = download_one(cand["url"], dest_dir, prefix, i)
        if p:
            downloaded += 1
            attribution = {
                **cand,
                "espece": scientific_name,
                "slug": slug,
                "fichier": str(p.relative_to(ROOT)),
                "date_telechargement": time.strftime("%Y-%m-%d"),
            }
            ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
            with ATTRIBUTION_PATH.open("a", encoding="utf-8") as manifest:
                manifest.write(json.dumps(attribution, ensure_ascii=False) + "\n")
        else:
            skipped += 1
        time.sleep(0.12)

    total = len(list(dest_dir.glob("*.jpg")))
    excluded = _manual_exclusions(slug)
    print(
        f"  {scientific_name:32s} | +{downloaded} (skip {skipped}) "
        f"| avant={existing} après={total} | exclus manuel={excluded}"
    )
    return {
        "espece": scientific_name,
        "slug": slug,
        "avant": existing,
        "telecharges": downloaded,
        "sautés": skipped,
        "apres": total,
        "exclus_manuellement": excluded,
        "candidats_sources": len(candidates),
    }


def main():
    p = argparse.ArgumentParser(description="Enrichissement du dataset ArboScan")
    p.add_argument("--per-species", type=int, default=60, help="Objectif images/espèce (cumul)")
    p.add_argument("--only", nargs="*", help="Limiter à certains slugs")
    p.add_argument("--sources", default="inat,wikimedia", help="inat,wikimedia")
    a = p.parse_args()
    sources = tuple(s.strip() for s in a.sources.split(",") if s.strip())

    df_rows = []
    if CSV_PATH.exists():
        import pandas as pd

        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        df_rows = df["nom_scientifique"].dropna().tolist()
    else:
        df_rows = [s.replace("_", " ") for s in discover_classes()]

    if a.only:
        wanted = set(a.only)
        df_rows = [n for n in df_rows if slugify(n) in wanted]

    print(f"Scraping {len(df_rows)} espèces (objectif {a.per_species}/espèce, sources={sources})\n")
    report = []
    for name in df_rows:
        report.append(scrape_species(name.strip(), a.per_species, sources))
        time.sleep(0.5)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    total_new = sum(r["telecharges"] for r in report)
    total_all = sum(r["apres"] for r in report)
    print(f"\nBilan : +{total_new} images téléchargées | total dataset = {total_all}")
    print(f"Rapport: {REPORT_PATH}")


if __name__ == "__main__":
    main()
