"""
Dataset and DataLoader setup for NIH CXR-14.

Supports:
  • Full dataset mode  – official train_val_list.txt / test_list.txt splits
  • Sample mode        – a tiny subset for fast pipeline validation

No horizontal flip (heart laterality), conservative crop ≥90%.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from config import (
    RAW_DIR, SAMPLE_DIR, DISEASE_LABELS, NUM_CLASSES,
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE,
)


# ── transforms ───────────────────────────────────────────────────────────────
def get_train_transforms() -> transforms.Compose:
    """Conservative augmentation — small crop, gentle rotation, no hflip."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),          # slight margin
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic centre-crop pipeline for val/test."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── dataset ──────────────────────────────────────────────────────────────────
class ChestXrayDataset(Dataset):
    """
    Returns (image_tensor, label_tensor) pairs.
    image: FloatTensor (3, H, W), labels: FloatTensor (14,) multi-hot.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_index: dict,
        transform: transforms.Compose,
    ):
        self.df = df[df["Image Index"].isin(image_index)].reset_index(drop=True)
        self.image_index = image_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = self.image_index[row["Image Index"]]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        labels = torch.tensor(
            row[DISEASE_LABELS].values.astype(np.float32),
            dtype=torch.float32,
        )
        return image, labels

    def get_image_path(self, idx: int) -> Path:
        """Return the filesystem path for a given dataset index."""
        fname = self.df.iloc[idx]["Image Index"]
        return self.image_index[fname]

    def get_metadata(self, idx: int) -> dict:
        """Return a dict of metadata for a given dataset index."""
        row = self.df.iloc[idx]
        return {
            "filename": row["Image Index"],
            "labels": [DISEASE_LABELS[i] for i, v in enumerate(
                row[DISEASE_LABELS].values) if v == 1],
            "patient_id": row.get("Patient ID", "unknown"),
        }


# ── helper functions ─────────────────────────────────────────────────────────
def _load_split_list(path: Path) -> set:
    """Read one filename per line; return a set."""
    return set(path.read_text().strip().splitlines())


def _parse_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary column for each of the 14 disease labels."""
    for label in DISEASE_LABELS:
        df[label] = df["Finding Labels"].str.contains(label, regex=False).astype(int)
    return df


def build_image_index(raw_dir: Path) -> dict:
    """
    Scan images_*/images/ subdirectories and return filename→Path map.
    Also handles flat directory layout (all PNGs in one folder).
    """
    index = {}
    # NIH layout: images_001/images/*.png, images_002/images/*.png, ...
    for img_dir in sorted(raw_dir.glob("images_*/images")):
        for p in img_dir.iterdir():
            if p.suffix.lower() == ".png":
                index[p.name] = p
    # Flat layout fallback: *.png directly in raw_dir
    if not index:
        for p in raw_dir.iterdir():
            if p.suffix.lower() == ".png":
                index[p.name] = p
    return index


def compute_pos_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """Per-class pos_weight = neg_count / pos_count for BCEWithLogitsLoss."""
    pos = train_df[DISEASE_LABELS].sum(axis=0).values.astype(np.float32)
    neg = len(train_df) - pos
    weights = neg / np.maximum(pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32)


# ── main builder ─────────────────────────────────────────────────────────────
def build_dataloaders(
    raw_dir: Path = RAW_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,       # 0 for Windows compat by default
    val_fraction: float = 0.1,
    use_weighted_sampler: bool = True,
    seed: int = 42,
    sample_n: Optional[int] = None,
) -> tuple:
    """
    Return (train_loader, val_loader, test_loader, pos_weights).

    If sample_n is set, take only that many rows from the full CSV
    for rapid pipeline testing.
    """
    rng = np.random.default_rng(seed)

    # ── load and annotate metadata ────────────────────────────────────────
    csv_path = raw_dir / "Data_Entry_2017.csv"
    if not csv_path.exists():
        # Try alternate common name
        csv_path = raw_dir / "Data_Entry_2017_v2020.csv"
    df = pd.read_csv(csv_path)
    df = _parse_labels(df)

    # ── optional sample ───────────────────────────────────────────────────
    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        print(f"[SAMPLE MODE] Using {sample_n} images out of {len(df)}")

    # ── apply official train / test split ─────────────────────────────────
    train_val_list_path = raw_dir / "train_val_list.txt"
    test_list_path = raw_dir / "test_list.txt"

    if train_val_list_path.exists() and test_list_path.exists():
        train_val_files = _load_split_list(train_val_list_path)
        test_files = _load_split_list(test_list_path)
        train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
        test_df = df[df["Image Index"].isin(test_files)].copy()
    else:
        # No split files → random 80/20
        print("[WARNING] No official split files found → using random 80/20 split")
        n_train_val = int(len(df) * 0.8)
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_val_df = shuffled.iloc[:n_train_val].copy()
        test_df = shuffled.iloc[n_train_val:].copy()

    # ── patient-level train / val split ───────────────────────────────────
    if "Patient ID" in train_val_df.columns:
        patients = train_val_df["Patient ID"].unique()
        rng.shuffle(patients)
        n_val = max(1, int(len(patients) * val_fraction))
        val_patients = set(patients[:n_val])

        train_df = train_val_df[~train_val_df["Patient ID"].isin(val_patients)].copy()
        val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)].copy()
    else:
        n_val = max(1, int(len(train_val_df) * val_fraction))
        val_df = train_val_df.iloc[:n_val].copy()
        train_df = train_val_df.iloc[n_val:].copy()

    print(f"Split sizes — train: {len(train_df):,}  "
          f"val: {len(val_df):,}  test: {len(test_df):,}")

    # ── image index ───────────────────────────────────────────────────────
    image_index = build_image_index(raw_dir)
    print(f"Images found on disk: {len(image_index):,}")

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = ChestXrayDataset(train_df, image_index, get_train_transforms())
    val_ds = ChestXrayDataset(val_df, image_index, get_val_transforms())
    test_ds = ChestXrayDataset(test_df, image_index, get_val_transforms())

    print(f"Dataset sizes — train: {len(train_ds):,}  "
          f"val: {len(val_ds):,}  test: {len(test_ds):,}")

    # ── weighted sampler ──────────────────────────────────────────────────
    sampler = None
    if use_weighted_sampler and len(train_ds) > 0:
        label_sums = train_ds.df[DISEASE_LABELS].sum(axis=1).values
        sample_weights = np.where(label_sums > 0, 1.0 / label_sums, 1.0)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )

    # ── data loaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    pos_weights = compute_pos_weights(train_ds.df) if len(train_ds) > 0 \
        else torch.ones(NUM_CLASSES)

    return train_loader, val_loader, test_loader, pos_weights


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loaders = build_dataloaders(batch_size=4, num_workers=0)
    train_loader, val_loader, test_loader, pw = loaders

    if len(train_loader) > 0:
        images, labels = next(iter(train_loader))
        print(f"\nBatch shapes — images: {images.shape}  labels: {labels.shape}")
        print(f"Image dtype/range: {images.dtype}  [{images.min():.2f}, {images.max():.2f}]")
        print(f"pos_weights: {pw.numpy().round(1)}")
