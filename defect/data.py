"""Indexing, splitting and loading of the NEU-DET steel surface defect images."""
import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "NEU-DET"
# Which train/val/test protocol every script uses; outputs of each go to their own folders.
#   block  - consecutive image numbers kept together (default, no near-duplicate leakage)
#   random - plain stratified random split (the common practice, kept for comparison)
SPLIT = os.environ.get("DEFECT_SPLIT", "block")
RESULTS = ROOT / "results" / SPLIT
MODELS = ROOT / "models" / SPLIT
FIGURES = ROOT / "figures" / SPLIT
SPLITS_CSV = RESULTS / "splits.csv"
BLOCK = 30
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_NAMES = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted surface",
    "rolled-in_scale": "Rolled-in scale",
    "scratches": "Scratches",
}


def read_boxes(xml_path):
    """Defect bounding boxes as (xmin, ymin, xmax, ymax) tuples."""
    root = ET.parse(xml_path).getroot()
    return [tuple(int(obj.find(f"bndbox/{k}").text) for k in ("xmin", "ymin", "xmax", "ymax"))
            for obj in root.iter("object")]


def build_index(data_dir=DATA_DIR):
    """One row per image: path, class, label id, defect boxes and a pixel hash."""
    rows = []
    for path in sorted((data_dir / "IMAGES").glob("*.jpg")):
        cls = path.stem.rsplit("_", 1)[0]
        img = load_gray(path)
        rows.append({
            "file": path.name,
            "cls": cls,
            "label": CLASSES.index(cls),
            "boxes": read_boxes(data_dir / "ANNOTATIONS" / f"{path.stem}.xml"),
            "md5": hashlib.md5(img.tobytes()).hexdigest(),
        })
    return pd.DataFrame(rows)


def make_splits(index, seed=42, val_frac=0.2, test_frac=0.2):
    """Stratified 60/20/20 split, fixed once and saved so every experiment shares it."""
    trainval, test = train_test_split(index, test_size=test_frac, stratify=index["label"], random_state=seed)
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  stratify=trainval["label"], random_state=seed)
    out = index.copy()
    out["split"] = "train"
    out.loc[val.index, "split"] = "val"
    out.loc[test.index, "split"] = "test"
    return out


def make_block_splits(index, block=BLOCK, seed=42):
    """Split whole blocks of consecutive image numbers (6/2/2 blocks of 30 per class).

    Neighbouring images are near-identical frames of the same steel strip; a random
    split would put siblings on both sides and inflate the test score.
    """
    rng = np.random.default_rng(seed)
    out = index.copy()
    out["block"] = (out["file"].str.extract(r"_(\d+)\.jpg")[0].astype(int) - 1) // block
    out["split"] = "train"
    for cls in CLASSES:
        blocks = rng.permutation(np.sort(out.loc[out["cls"] == cls, "block"].unique()))
        n = len(blocks)
        n_val = n_test = round(n * 0.2)
        out.loc[(out["cls"] == cls) & out["block"].isin(blocks[:n_test]), "split"] = "test"
        out.loc[(out["cls"] == cls) & out["block"].isin(blocks[n_test:n_test + n_val]), "split"] = "val"
    return out


def write_splits(index):
    """Save the split for the active protocol; exact duplicates are dropped first."""
    index = index.drop_duplicates("md5")
    split = make_block_splits(index) if SPLIT == "block" else make_splits(index)
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in ("file", "cls", "label", "block", "split") if c in split.columns]
    split[keep].to_csv(SPLITS_CSV, index=False)
    return split


def load_splits():
    """Read the saved split, creating it on first use."""
    if not SPLITS_CSV.exists():
        write_splits(build_index())
    return pd.read_csv(SPLITS_CSV)


def load_gray(path):
    """The JPEGs are stored as RGB but the content is grayscale."""
    return np.asarray(Image.open(path).convert("L"))
