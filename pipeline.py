"""
Full end-to-end pipeline orchestrator.

Implements the complete flow:
  ① Input image
  ② Stage 1 — EfficientNet-B4 fast screening + temperature scaling
  ③ Routing decision — calibrated p ≥ τ?
  ④ Stage 2 — VLM second opinion (if needed)
  ⑤ Reconciliation layer — weighted fusion
  ⑥ Final decision + explanation
  ⑦ Feedback store integration
  ⑧ Calibration update from feedback

Usage:
    from pipeline import DiagnosticPipeline
    pipe = DiagnosticPipeline()
    result = pipe.analyze("path/to/xray.png")
    print(result.to_dict())
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from config import (
    CKPT_DIR, DISEASE_LABELS, NUM_CLASSES,
    IMAGE_SIZE, ROUTING_THRESHOLD,
    CALIBRATION_UPDATE_INTERVAL,
    get_device,
)
from dataset import get_val_transforms
from stage1_model import build_stage1_model, tune_temperature
from stage2_vlm import Stage2Engine
from reconciliation import ReconciliationLayer, PipelineResult
from feedback_store import FeedbackStore


class DiagnosticPipeline:
    """
    Two-stage chest X-ray diagnostic pipeline.

    Flow:
      1. Preprocess image
      2. Stage 1: EfficientNet-B4 → calibrated probability
      3. Route: if max(calibrated_prob) ≥ τ → skip Stage 2
      4. Stage 2: VLM analysis (if routed)
      5. Reconcile both stage outputs
      6. Return structured result with explanation
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        threshold: float = ROUTING_THRESHOLD,
        try_vlm: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.threshold = threshold
        self.transform = get_val_transforms()

        # ── Stage 1 model ────────────────────────────────────────────────
        self.model = build_stage1_model(pretrained=True).to(self.device)

        # Try to load trained checkpoint
        if checkpoint_path is None:
            # Auto-detect best checkpoint
            for name in ["best_model_calibrated.pth", "best_model.pth"]:
                p = CKPT_DIR / name
                if p.exists():
                    checkpoint_path = p
                    break

        if checkpoint_path and checkpoint_path.exists():
            print(f"[Pipeline] Loading checkpoint: {checkpoint_path.name}")
            ckpt = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            T = ckpt.get("temperature", 1.5)
            self.model.temp_scaler.log_temperature.data = torch.log(
                torch.tensor(T, dtype=torch.float32)
            )
            print(f"[Pipeline] Loaded (epoch {ckpt.get('epoch', '?')}, "
                  f"T={T:.4f})")
        else:
            print("[Pipeline] WARNING: No trained checkpoint found. "
                  "Using untrained model (random head weights).")

        self.model.eval()

        # ── Stage 2 engine ───────────────────────────────────────────────
        print("[Pipeline] Initialising Stage 2 …")
        self.stage2 = Stage2Engine(try_vlm=try_vlm)

        # ── Reconciliation ───────────────────────────────────────────────
        self.reconciler = ReconciliationLayer()

        # ── Feedback store ───────────────────────────────────────────────
        self.feedback = FeedbackStore()

        # Track pipeline statistics
        self._stats = {
            "total_cases": 0,
            "stage2_invoked": 0,
            "conflicts": 0,
            "review_flagged": 0,
        }

        print(f"[Pipeline] Ready. Threshold τ={self.threshold}, "
              f"Device={self.device}, "
              f"T={self.model.temp_scaler.temperature.item():.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # Core analysis
    # ══════════════════════════════════════════════════════════════════════
    def analyze(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        image_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Analyse a single chest X-ray through the full pipeline.

        Args:
            image_input: path, PIL Image, or numpy array
            image_id: optional identifier for tracking

        Returns:
            PipelineResult with all stage outputs and final decision
        """
        self._stats["total_cases"] += 1

        # ── ① Preprocess ─────────────────────────────────────────────────
        image_pil, image_path = self._load_image(image_input)
        if image_id is None:
            image_id = Path(image_path).name if image_path else f"case_{self._stats['total_cases']}"

        tensor = self.transform(image_pil).unsqueeze(0).to(self.device)

        # ── ② Stage 1: Fast screening ────────────────────────────────────
        with torch.no_grad():
            scaled_logits = self.model(tensor)                    # (1, 14)
            raw_logits = self.model.forward_logits(tensor)        # (1, 14) unscaled
            probs = torch.sigmoid(scaled_logits).cpu().numpy()[0]  # (14,)

        max_prob = float(np.max(probs))
        temperature = self.model.temp_scaler.temperature.item()

        # ── ③ Routing decision ───────────────────────────────────────────
        route_to_stage2 = max_prob < self.threshold

        # ── ④ Stage 2 (if needed) ────────────────────────────────────────
        stage2_result = None
        if route_to_stage2:
            self._stats["stage2_invoked"] += 1
            print(f"  [Route] max_prob={max_prob:.3f} < τ={self.threshold} "
                  f"→ invoking Stage 2")

            stage2_result = self.stage2.analyze(
                image=image_pil,
                logits=raw_logits.cpu().numpy()[0],
                probs=probs,
            )
        else:
            print(f"  [Route] max_prob={max_prob:.3f} ≥ τ={self.threshold} "
                  f"→ Stage 1 accepted directly")

        # ── ⑤ Reconciliation ─────────────────────────────────────────────
        result = self.reconciler.reconcile(
            stage1_probs=probs,
            stage2_result=stage2_result,
            routed_to_stage2=route_to_stage2,
            temperature=temperature,
            image_id=image_id,
            image_path=str(image_path) if image_path else "",
        )

        # ── Update stats ─────────────────────────────────────────────────
        if result.conflict_detected:
            self._stats["conflicts"] += 1
        if result.needs_radiologist_review:
            self._stats["review_flagged"] += 1

        return result

    # ══════════════════════════════════════════════════════════════════════
    # Batch analysis
    # ══════════════════════════════════════════════════════════════════════
    def analyze_batch(
        self,
        image_paths: list,
        progress: bool = True,
    ) -> list[PipelineResult]:
        """Analyse multiple images sequentially."""
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            if progress:
                print(f"\n[{i+1}/{total}] Analysing {Path(path).name} …")
            result = self.analyze(path)
            results.append(result)

        # ── Summary ──────────────────────────────────────────────────────
        if progress:
            print(f"\n{'═' * 60}")
            print(f"Batch complete: {total} images")
            print(f"  Stage 2 invoked : {self._stats['stage2_invoked']}")
            print(f"  Conflicts       : {self._stats['conflicts']}")
            print(f"  Review flagged  : {self._stats['review_flagged']}")
            print(f"{'═' * 60}")

        return results

    # ══════════════════════════════════════════════════════════════════════
    # ⑦ Feedback — radiologist correction
    # ══════════════════════════════════════════════════════════════════════
    def submit_correction(
        self,
        result: PipelineResult,
        radiologist_finding: str,
        radiologist_severity: str = "unknown",
        notes: str = "",
        true_labels: Optional[list] = None,
    ) -> None:
        """
        Record a radiologist correction and check if calibration
        update is needed.
        """
        self.feedback.add_from_dict(
            image_id=result.image_id,
            pipeline_finding=result.final_finding,
            pipeline_confidence=result.final_confidence,
            radiologist_finding=radiologist_finding,
            radiologist_severity=radiologist_severity,
            was_routed_to_stage2=result.routed_to_stage2,
            conflict_detected=result.conflict_detected,
            notes=notes,
            stage1_logits=result.stage1_probs.tolist(),  # store for recalibration
            stage1_probs=result.stage1_probs.tolist(),
            true_labels=true_labels,
        )

        # ── ⑧ Check if calibration update needed ────────────────────────
        n_corrections = self.feedback.count()
        if n_corrections % CALIBRATION_UPDATE_INTERVAL == 0:
            print(f"\n[Pipeline] {n_corrections} corrections accumulated "
                  f"→ triggering calibration update")
            self._update_calibration()

    def _update_calibration(self) -> None:
        """
        Retune temperature scaling from accumulated feedback.
        Lightweight continuous improvement without full retraining.
        """
        logits, labels = self.feedback.get_calibration_data()
        if logits is None or len(logits) < 10:
            print("[Pipeline] Not enough calibration data yet "
                  f"(need ≥10, have {len(logits) if logits is not None else 0})")
            return

        # Create a simple DataLoader from the feedback data
        from torch.utils.data import TensorDataset, DataLoader

        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)
        cal_dataset = TensorDataset(logits_t, labels_t)
        cal_loader = DataLoader(cal_dataset, batch_size=64, shuffle=False)

        # We need a special loader that yields (images, labels) but we
        # only have logits. Use the tune_temperature function with a
        # custom approach.
        old_T = self.model.temp_scaler.temperature.item()

        # Simple grid search for temperature on stored data
        best_T, best_loss = old_T, float("inf")
        criterion = torch.nn.BCEWithLogitsLoss()

        for T_candidate in np.arange(0.5, 3.0, 0.05):
            scaled = logits_t / T_candidate
            loss = criterion(scaled, labels_t).item()
            if loss < best_loss:
                best_loss = loss
                best_T = T_candidate

        self.model.temp_scaler.log_temperature.data = torch.log(
            torch.tensor(best_T, dtype=torch.float32)
        )

        print(f"[Calibration] T updated: {old_T:.4f} → {best_T:.4f} "
              f"(loss: {best_loss:.4f})")

    # ══════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════
    def _load_image(self, image_input) -> tuple:
        """Normalise input to (PIL.Image, path_str)."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            return Image.open(path).convert("RGB"), str(path)
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB"), None
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(image_input).convert("RGB"), None
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def get_stats(self) -> dict:
        """Return pipeline statistics."""
        return {
            **self._stats,
            "temperature": self.model.temp_scaler.temperature.item(),
            "feedback_count": self.feedback.count(),
            "feedback_stats": self.feedback.get_statistics(),
        }

    def update_threshold(self, new_threshold: float) -> None:
        """Dynamically adjust routing threshold."""
        old = self.threshold
        self.threshold = new_threshold
        print(f"[Pipeline] Threshold updated: {old:.3f} → {new_threshold:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI — run pipeline on a single image or directory
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CXR diagnostic pipeline")
    parser.add_argument("input", type=str,
                        help="Path to an image or directory of images")
    parser.add_argument("--threshold", type=float, default=ROUTING_THRESHOLD,
                        help=f"Routing threshold τ (default: {ROUTING_THRESHOLD})")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM (use rule-based fallback)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")

    args = parser.parse_args()

    pipe = DiagnosticPipeline(
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        threshold=args.threshold,
        try_vlm=not args.no_vlm,
    )

    input_path = Path(args.input)
    if input_path.is_file():
        result = pipe.analyze(input_path)
        print("\n" + json.dumps(result.to_dict(), indent=2, default=str))
    elif input_path.is_dir():
        images = sorted(input_path.glob("*.png"))
        if not images:
            images = sorted(input_path.glob("*.jpg"))
        results = pipe.analyze_batch(images)
        for r in results:
            print(f"\n{r.image_id}: {r.final_finding} "
                  f"(conf={r.final_confidence:.3f}, "
                  f"review={r.needs_radiologist_review})")
    else:
        print(f"Error: {input_path} not found")
