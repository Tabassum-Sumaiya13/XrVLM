"""
run_sample.py — End-to-end pipeline with REAL NIH Chest X-ray data.

This script:
  1. Downloads real CXR images from Kaggle (first batch, ~10k images)
  2. Trains Stage 1 (EfficientNet-B4) on the real data
  3. Calibrates temperature scaling
  4. Runs the full two-stage pipeline on test images
  5. Simulates radiologist feedback
  6. Demonstrates calibration update from feedback

Usage:
    python run_sample.py                              # download + train + run
    python run_sample.py --skip-download              # skip Kaggle download
    python run_sample.py --skip-training              # skip training too
    python run_sample.py --n-images 500               # limit to 500 images
    python run_sample.py --epochs-s1 3 --epochs-s2 12 # full training
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from config import RAW_DIR, CKPT_DIR, OUTPUT_DIR, DISEASE_LABELS, get_device


def step_banner(title: str, step_num: int):
    """Print a formatted step banner."""
    icons = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧"}
    icon = icons.get(step_num, f"[{step_num}]")
    print(f"\n{'━' * 70}")
    print(f"  {icon}  {title}")
    print(f"{'━' * 70}\n")


def count_available_images(raw_dir: Path) -> int:
    """Count real images in the data directory."""
    count = 0
    for img_dir in raw_dir.glob("images_*/images"):
        count += len(list(img_dir.glob("*.png")))
    count += len(list(raw_dir.glob("*.png")))
    return count


def main(
    n_images: int = None,
    skip_download: bool = False,
    skip_training: bool = False,
    train_epochs_s1: int = 2,
    train_epochs_s2: int = 3,
    batch_size: int = 8,
    threshold: float = 0.95,
    n_batches: int = 1,
):
    t_start = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  Two-Stage CXR Diagnostic Pipeline — REAL DATA Run                ║")
    print("╚" + "═" * 68 + "╝")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Download real NIH Chest X-ray data from Kaggle", 1)
    # ══════════════════════════════════════════════════════════════════════

    if not skip_download:
        from download_sample import download_real_data
        success = download_real_data(
            dest_dir=RAW_DIR,
            n_batches=n_batches,
            n_limit=n_images,
        )
        if not success:
            print("\n" + "!" * 70)
            print("  DOWNLOAD FAILED — Cannot proceed without real data.")
            print("  Check Kaggle setup:")
            print("    1. pip install kaggle")
            print("    2. Place kaggle.json in ~/.kaggle/")
            print("    3. Accept dataset rules at:")
            print(f"       https://www.kaggle.com/datasets/nih-chest-xrays/data")
            print("!" * 70)
            sys.exit(1)
    else:
        print("Skipping download (--skip-download flag)")

    # Verify we have real images
    n_available = count_available_images(RAW_DIR)
    csv_exists = (RAW_DIR / "Data_Entry_2017.csv").exists()
    print(f"\nData verification:")
    print(f"  Images on disk : {n_available:,}")
    print(f"  Metadata CSV   : {'YES' if csv_exists else 'NO'}")

    if n_available == 0 or not csv_exists:
        print("\nERROR: No data available. Run without --skip-download first.")
        sys.exit(1)

    sample_n = n_images if n_images else None

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Train Stage 1 — EfficientNet-B4 on real CXR data", 2)
    # ══════════════════════════════════════════════════════════════════════

    if not skip_training:
        from train import train
        best_ckpt = train(
            epochs_stage1=train_epochs_s1,
            epochs_stage2=train_epochs_s2,
            batch_size=batch_size,
            num_workers=0,
            sample_n=sample_n,
        )
        print(f"\nTraining complete. Best checkpoint: {best_ckpt}")
    else:
        print("Skipping training (--skip-training flag)")
        best_ckpt = CKPT_DIR / "best_model_calibrated.pth"
        if not best_ckpt.exists():
            best_ckpt = CKPT_DIR / "best_model.pth"
        if not best_ckpt.exists():
            print("ERROR: No checkpoint found. Run without --skip-training first.")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Routing decision — Calibrated p >= threshold", 3)
    # ══════════════════════════════════════════════════════════════════════

    print(f"Routing threshold τ = {threshold}")
    print(f"  max(calibrated_prob) >= {threshold} → Stage 1 accepts (skip VLM)")
    print(f"  max(calibrated_prob) <  {threshold} → Route to Stage 2 (VLM)\n")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Run full pipeline on real test images", 4)
    # ══════════════════════════════════════════════════════════════════════

    from pipeline import DiagnosticPipeline

    pipe = DiagnosticPipeline(
        checkpoint_path=best_ckpt if best_ckpt and Path(best_ckpt).exists() else None,
        threshold=threshold,
        try_vlm=False,     # use rule-based fallback (CheXagent needs ~8GB VRAM)
    )

    # Get test images from the dataset
    from dataset import build_image_index
    image_index = build_image_index(RAW_DIR)
    test_list_path = RAW_DIR / "test_list.txt"

    if test_list_path.exists():
        test_filenames = test_list_path.read_text().strip().splitlines()[:20]
        test_images = [image_index[f] for f in test_filenames if f in image_index]
    else:
        # Fallback: grab some images directly
        all_images = sorted(image_index.values())
        test_images = all_images[-20:]  # last 20

    n_test = min(len(test_images), 20)
    test_images = test_images[:n_test]
    print(f"Analysing {n_test} real test images ...\n")

    results = []
    for img_path in test_images:
        result = pipe.analyze(img_path)
        results.append(result)
        print(f"  {result.image_id}: "
              f"finding={result.final_finding}, "
              f"confidence={result.final_confidence:.3f}, "
              f"routed_s2={result.routed_to_stage2}, "
              f"review={result.needs_radiologist_review}")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Reconciliation summary", 5)
    # ══════════════════════════════════════════════════════════════════════

    stage1_only = sum(1 for r in results if not r.routed_to_stage2)
    stage2_used = sum(1 for r in results if r.routed_to_stage2)
    conflicts = sum(1 for r in results if r.conflict_detected)
    reviews = sum(1 for r in results if r.needs_radiologist_review)

    print(f"Total cases      : {len(results)}")
    print(f"Stage 1 accepted : {stage1_only} ({100*stage1_only/max(len(results),1):.0f}%)")
    print(f"Stage 2 invoked  : {stage2_used} ({100*stage2_used/max(len(results),1):.0f}%)")
    print(f"Conflicts        : {conflicts}")
    print(f"Review flagged   : {reviews}")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Final decision + explanation (sample cases)", 6)
    # ══════════════════════════════════════════════════════════════════════

    for i, r in enumerate(results[:5]):
        print(f"\n{'─' * 60}")
        print(f"Case {i+1}: {r.image_id}")
        print(f"  Finding     : {r.final_finding}")
        print(f"  Severity    : {r.final_severity}")
        print(f"  Confidence  : {r.final_confidence:.3f}")
        print(f"  Temperature : {r.temperature:.4f}")
        explanation = r.explanation
        if len(explanation) > 300:
            explanation = explanation[:300] + "..."
        print(f"  Explanation : {explanation}")
        if r.secondary_findings:
            print(f"  Secondary   : {', '.join(r.secondary_findings[:5])}")
        if r.needs_radiologist_review:
            print(f"  ⚠ FLAGGED FOR RADIOLOGIST REVIEW")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Feedback store — simulate radiologist corrections", 7)
    # ══════════════════════════════════════════════════════════════════════

    print("Simulating radiologist reviews ...\n")

    for i, r in enumerate(results[:5]):
        # Simulate: radiologist agrees 60% of time, corrects 40%
        if i % 3 == 0:
            radiologist_finding = "Effusion"  # simulated disagreement
        else:
            radiologist_finding = r.final_finding  # agreement

        pipe.submit_correction(
            result=r,
            radiologist_finding=radiologist_finding,
            radiologist_severity="moderate",
            notes=f"Simulated review #{i+1}",
            true_labels=[0] * 14,
        )
        match = "AGREE" if r.final_finding == radiologist_finding else "DISAGREE"
        print(f"  Case {r.image_id}: pipeline={r.final_finding}, "
              f"radiologist={radiologist_finding} [{match}]")

    fb_stats = pipe.feedback.get_statistics()
    print(f"\nFeedback stats:")
    print(f"  Total corrections : {fb_stats['total_corrections']}")
    print(f"  Pipeline accuracy : {fb_stats['accuracy']:.1%}")

    # ══════════════════════════════════════════════════════════════════════
    step_banner("Calibration update (from feedback)", 8)
    # ══════════════════════════════════════════════════════════════════════

    T_current = pipe.model.temp_scaler.temperature.item()
    print(f"Current temperature    : {T_current:.4f}")
    print(f"Corrections stored     : {pipe.feedback.count()}")
    print(f"Auto-update interval   : every 50 corrections")
    print(f"(T will be automatically retuned as more corrections accumulate)")

    # ══════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start

    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║'}  Pipeline run complete!{'':>45}{'║'}")
    print(f"{'╠' + '═' * 68 + '╣'}")
    print(f"{'║'}  Total time        : {elapsed:.1f}s{'':>{47 - len(f'{elapsed:.1f}s')}}{'║'}")
    print(f"{'║'}  Images analysed   : {len(results)}{'':>{47 - len(str(len(results)))}}{'║'}")
    print(f"{'║'}  Real data         : YES (NIH CXR-14 from Kaggle){'':>19}{'║'}")
    print(f"{'║'}  Stage 2 rate      : {100*stage2_used/max(len(results),1):.0f}%{'':>{46 - len(f'{100*stage2_used/max(len(results),1):.0f}%')}}{'║'}")
    print(f"{'║'}  Temperature       : {T_current:.4f}{'':>{47 - len(f'{T_current:.4f}')}}{'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}")

    # Save results
    results_path = OUTPUT_DIR / "sample_results.json"
    with open(results_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline test with REAL NIH data"
    )
    parser.add_argument("--n-images", type=int, default=None,
                        help="Limit total images from dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip Kaggle download (use existing data)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training (use existing checkpoint)")
    parser.add_argument("--epochs-s1", type=int, default=2,
                        help="Stage 1 warmup epochs")
    parser.add_argument("--epochs-s2", type=int, default=3,
                        help="Stage 2 finetune epochs")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--batches", type=int, default=1,
                        help="Number of Kaggle image batches to download (1-12)")

    args = parser.parse_args()

    main(
        n_images=args.n_images,
        skip_download=args.skip_download,
        skip_training=args.skip_training,
        train_epochs_s1=args.epochs_s1,
        train_epochs_s2=args.epochs_s2,
        batch_size=args.batch,
        threshold=args.threshold,
        n_batches=args.batches,
    )
