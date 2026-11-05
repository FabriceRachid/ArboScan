"""ArboScan — dataset : split stratifié, Dataset PyTorch, augmentation."""
from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = ROOT / "model" / "data" / "images"
METADATA_DIR = ROOT / "model" / "data" / "metadata"
CURATED_DIR = ROOT / "model" / "data" / "curated"
CSV_PATH = METADATA_DIR / "species.csv"

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.strip().lower()).strip("_")


def discover_classes(samples_dir: Path = SAMPLES_DIR) -> dict[str, Path]:
    """Retourne {slug_espece: dossier} pour chaque classe ayant au moins 1 image."""
    classes = {}
    if not samples_dir.exists():
        return classes
    for d in sorted(samples_dir.iterdir()):
        if not d.is_dir():
            continue
        n = sum(1 for f in d.iterdir() if f.suffix.lower() in VALID_EXT)
        if n > 0:
            classes[d.name] = d
    return classes


def stratified_split(
    classes: dict[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Split 70/15/15 par espèce en gardant ensemble les images d'une même source."""
    rng = random.Random(seed)
    image_groups = _load_image_groups()
    splits: dict[str, dict[str, list[str]]] = {}
    for slug, folder in classes.items():
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in VALID_EXT)
        rng.shuffle(files)
        n = len(files)
        n_train = max(1, int(n * train_ratio))
        n_val = int(n * val_ratio)
        if n_train + n_val >= n and n > 2:
            n_val = max(1, n - n_train - 1) if n - n_train - 1 >= 1 else 0
        n_test = n - n_train - n_val
        if n_test <= 0 and n > 1:
            n_test = 1
            n_train = max(1, n - n_val - n_test)
        initial = {
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val : n_train + n_val + n_test],
        }
        original_bucket = {name: bucket for bucket, names in initial.items() for name in names}
        groups: dict[str, list[str]] = defaultdict(list)
        for name in files:
            groups[image_groups.get((slug, name), f"single:{name}")].append(name)

        grouped = {key: [] for key in ("train", "val", "test")}
        current_sizes = Counter()
        targets = {"train": n * train_ratio, "val": n * val_ratio, "test": n * (1 - train_ratio - val_ratio)}
        group_items = list(groups.items())
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        for _, names in group_items:
            votes = Counter(original_bucket[name] for name in names)
            max_votes = max(votes.values())
            tied = [bucket for bucket in ("train", "val", "test") if votes[bucket] == max_votes]
            bucket = min(tied, key=lambda key: current_sizes[key] / max(targets[key], 1))
            grouped[bucket].extend(names)
            current_sizes[bucket] += len(names)
        splits[slug] = grouped
    return splits


def _load_image_groups() -> dict[tuple[str, str], str]:
    """Regroupe les vues d'une observation iNaturalist et les séries Commons."""
    manifest = METADATA_DIR / "image_sources.jsonl"
    if not manifest.exists():
        return {}
    groups = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("review_status") == "excluded_manual":
            continue
        file_name = Path(str(row.get("fichier", "")).replace("\\", "/")).name
        slug = row.get("slug")
        if not file_name or not slug:
            continue
        if row.get("source") == "inaturalist":
            observation_id = str(row.get("page", "")).rstrip("/").split("/")[-1]
            group = f"inat:{observation_id}" if observation_id.isdigit() else f"single:{file_name}"
        else:
            title = str(row.get("page", file_name)).removeprefix("File:")
            stem = Path(title).stem.lower()
            stem = re.sub(r"\s*\((?:\d+(?:\s+of\s+\d+)?)\)\s*$", "", stem)
            stem = re.sub(r"\s+\d+$", "", stem)
            stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
            stem = re.sub(r"[^a-z0-9]+", " ", stem).strip()
            group = f"commons:{slug}:{stem or file_name}"
        groups[(slug, file_name)] = group
    return groups


def save_splits(splits: dict, path: Path | None = None) -> Path:
    path = path or (METADATA_DIR / "splits.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_splits(path: Path | None = None) -> dict:
    path = path or (METADATA_DIR / "splits.json")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable — lance d'abord : python -m model.src.dataset --prepare"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_labels(splits: dict | None = None) -> dict[str, int]:
    """Mapping slug -> index numérique, trié pour être stable."""
    splits = splits or load_splits()
    slugs = sorted(splits.keys())
    return {slug: i for i, slug in enumerate(slugs)}


def load_metadata() -> pd.DataFrame:
    """Charge especes.csv et ajoute une colonne 'slug'."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=["nom_scientifique", "slug"])
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df["slug"] = df["nom_scientifique"].map(slugify)
    return df


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.2),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(IMG_SIZE * 1.15)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TreeLeafFruitDataset(Dataset):
    """Dataset image + label à partir du split JSON (chemins relatifs à model/data/images/)."""

    def __init__(
        self,
        samples_dir: Path,
        split: dict[str, list[str]],
        labels: dict[str, int],
        train: bool = True,
        transform=None,
    ):
        self.samples_dir = samples_dir
        self.labels = labels
        self.transform = transform or get_transforms(train)
        self.items: list[tuple[Path, int]] = []
        for slug, files in split.items():
            if slug not in labels:
                continue
            for fname in files:
                p = samples_dir / slug / fname
                if p.exists():
                    self.items.append((p, labels[slug]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loaders(
    samples_dir: Path = SAMPLES_DIR,
    splits: dict | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    splits = splits or load_splits()
    labels = build_labels(splits)
    train_ds = TreeLeafFruitDataset(samples_dir, _pick(splits, "train"), labels, train=True)
    val_ds = TreeLeafFruitDataset(samples_dir, _pick(splits, "val"), labels, train=False)
    test_ds = TreeLeafFruitDataset(samples_dir, _pick(splits, "test"), labels, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, labels


def _pick(splits: dict, key: str) -> dict[str, list[str]]:
    """Adapte le format {slug: {train:[], val:[], test:[]}} -> {slug: [fichiers]}."""
    return {slug: parts.get(key, []) for slug, parts in splits.items() if parts.get(key)}


def prepare(samples_dir: Path = SAMPLES_DIR) -> dict:
    classes = discover_classes(samples_dir)
    if not classes:
        raise RuntimeError(f"Aucune image trouvée dans {samples_dir}")
    splits = stratified_split(classes)
    save_splits(splits)
    labels = build_labels(splits)
    n = {k: sum(len(v[k]) for v in splits.values()) for k in ("train", "val", "test")}
    print(f"Classes : {len(labels)} | images : train={n['train']} val={n['val']} test={n['test']}")
    faibles = {
        s: sum(len(files) for files in parts.values())
        for s, parts in splits.items()
        if sum(len(files) for files in parts.values()) < 15
    }
    if faibles:
        print("[!] Espèces avec <15 images (performance limitée) :")
        for s, c in sorted(faibles.items(), key=lambda x: x[1]):
            print(f"   - {s}: {c}")
    return splits


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Prépare le dataset ArboScan")
    p.add_argument("--prepare", action="store_true", help="Crée le split stratifié")
    p.add_argument("--samples", default=str(SAMPLES_DIR))
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if args.prepare:
        prepare(Path(args.samples))
    else:
        tl, vl, te, labels = make_loaders(Path(args.samples), batch_size=args.batch_size)
        print(f"Labels : {len(labels)} classes | train={len(tl.dataset)} val={len(vl.dataset)} test={len(te.dataset)}")
