"""
Download REAL NIH Chest X-ray data using Hugging Face Datasets STREAMING.

This allows pulling a specific number of real samples without downloading 
the entire 45GB dataset.

Usage:
    python download_sample.py --n 100          # Get 100 real samples
"""

import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from config import RAW_DIR, DISEASE_LABELS


def download_real_data(
    dest_dir: Path = RAW_DIR,
    n_batches: int = 1,
    n_limit: int = None,
) -> bool:
    """
    Backwards-compatible helper used by `run_sample.py`.

    The repo originally described a Kaggle-based downloader; the current
    implementation streams from Hugging Face instead.

    Args:
        dest_dir: Destination directory (typically `data/raw`).
        n_batches: Kept for API compatibility. Used only to scale a default
            when `n_limit` is not provided.
        n_limit: Max number of images to download.

    Returns:
        True on success, False on failure.
    """
    try:
        if n_limit is None:
            # Historically, one "batch" was ~10k images. Keep a similar scale.
            n_limit = int(max(1, n_batches)) * 10_000
        download_streamed_data(n_samples=int(n_limit), dest_dir=dest_dir)
        return True
    except Exception as e:
        print(f"[download_real_data] ERROR: {e}")
        return False

def download_streamed_data(n_samples: int = 100, dest_dir: Path = RAW_DIR):
    """
    Stream data from Hugging Face and save locally.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_dir = dest_dir # Keep it flat for simple loading
    
    print(f"\n{'═' * 70}")
    print(f"  NIH CHEST X-RAY — STREAMING FROM HUGGING FACE")
    print(f"{'═' * 70}")
    print(f"  Target: {n_samples} real images")
    print(f"  Dest:   {dest_dir}")
    print(f"{'═' * 70}\n")

    # BahaaEldin0/NIH-Chest-Xray-14 is a common mirror
    # We use streaming=True to avoid downloading the whole thing
    print("Connecting to Hugging Face dataset stream...")
    try:
        dataset = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split='train', streaming=True)
    except Exception as e:
        print(f"Error loading BahaaEldin0: {e}")
        print("Trying alternative: Manas2703/chest-xray-14")
        dataset = load_dataset("Manas2703/chest-xray-14", split='train', streaming=True)

    metadata = []
    
    print(f"Streaming and saving {n_samples} samples...")
    
    # Iterate through the stream
    for i, sample in tqdm(enumerate(dataset), total=n_samples):
        if i >= n_samples:
            break
            
        # sample usually contains 'image' (PIL) and other metadata
        img = sample['image']
        
        # Determine image index/name
        # Some datasets have 'image_name', some we just index
        img_name = sample.get('image_name', f"{i:08d}_000.png")
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        img_path = images_dir / img_name
        
        # Save image
        img.save(img_path)
        
        # Map findings
        # Some HF datasets have a list of labels, others a string
        findings = sample.get('findings', 'No Finding')
        if isinstance(findings, list):
            findings = "|".join(findings)
            
        metadata.append({
            "Image Index": img_name,
            "Finding Labels": findings,
            "Follow-up #": sample.get('follow_up_number', 0),
            "Patient ID": sample.get('patient_id', i),
            "Patient Age": sample.get('patient_age', "050Y"),
            "Patient Gender": sample.get('patient_gender', "M"),
            "View Position": sample.get('view_position', "PA"),
            "OriginalImageWidth": 1024,
            "OriginalImageHeight": 1024
        })

    # Save CSV
    df = pd.DataFrame(metadata)
    csv_path = dest_dir / "Data_Entry_2017.csv"
    df.to_csv(csv_path, index=False)
    
    # Create split files
    filenames = df["Image Index"].tolist()
    n_train = int(len(filenames) * 0.8)
    train_val = filenames[:n_train]
    test = filenames[n_train:]

    (dest_dir / "train_val_list.txt").write_text("\n".join(train_val) + "\n")
    (dest_dir / "test_list.txt").write_text("\n".join(test) + "\n")

    print(f"\n{'═' * 70}")
    print(f"  ✓ STREAMING COMPLETE")
    print(f"  Saved {len(df)} images and metadata to {dest_dir}")
    print(f"{'═' * 70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream real NIH data from Hugging Face")
    parser.add_argument("--n", type=int, default=100, help="Number of samples")
    parser.add_argument("--dest", type=str, default=str(RAW_DIR))
    args = parser.parse_args()

    download_streamed_data(n_samples=args.n, dest_dir=Path(args.dest))