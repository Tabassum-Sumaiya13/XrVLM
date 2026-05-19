"""
Reconciliation layer — weighted fusion of Stage 1 and Stage 2 outputs.

Combines:
  • Stage 1 calibrated probability (fast CNN)
  • Stage 2 VLM verdict (structured JSON)
  • Conflict detection → radiologist review flag

If the two stages disagree beyond CONFLICT_THRESHOLD, the disagreement
itself becomes a signal that triggers a human review.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from config import (
    DISEASE_LABELS, NUM_CLASSES,
    RECON_WEIGHT_STAGE1, RECON_WEIGHT_STAGE2,
    CONFLICT_THRESHOLD, ROUTING_THRESHOLD,
)


@dataclass
class PipelineResult:
    """Final output of the two-stage pipeline for a single image."""

    # ── identification ───────────────────────────────────────────────────
    image_id: str = ""
    image_path: str = ""

    # ── Stage 1 outputs ──────────────────────────────────────────────────
    stage1_probs: np.ndarray = field(default_factory=lambda: np.zeros(14))
    stage1_top_finding: str = ""
    stage1_top_prob: float = 0.0
    stage1_max_prob: float = 0.0          # max across all 14 classes
    temperature: float = 1.0

    # ── Routing ──────────────────────────────────────────────────────────
    routed_to_stage2: bool = False
    routing_threshold: float = ROUTING_THRESHOLD

    # ── Stage 2 outputs (if invoked) ─────────────────────────────────────
    stage2_result: Optional[dict] = None
    stage2_finding: str = ""
    stage2_confidence: float = 0.0
    stage2_rationale: str = ""

    # ── Reconciled output ────────────────────────────────────────────────
    final_finding: str = ""
    final_confidence: float = 0.0
    final_severity: str = "unknown"
    explanation: str = ""
    secondary_findings: list = field(default_factory=list)

    # ── Flags ────────────────────────────────────────────────────────────
    needs_radiologist_review: bool = False
    conflict_detected: bool = False
    conflict_detail: str = ""

    def to_dict(self) -> dict:
        """Serialise for JSON / database storage."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "stage1": {
                "top_finding": self.stage1_top_finding,
                "top_prob": round(self.stage1_top_prob, 4),
                "max_prob": round(self.stage1_max_prob, 4),
                "all_probs": {
                    DISEASE_LABELS[i]: round(float(self.stage1_probs[i]), 4)
                    for i in range(NUM_CLASSES)
                },
                "temperature": round(self.temperature, 4),
            },
            "routing": {
                "routed_to_stage2": self.routed_to_stage2,
                "threshold": self.routing_threshold,
            },
            "stage2": {
                "invoked": self.routed_to_stage2,
                "finding": self.stage2_finding,
                "confidence": round(self.stage2_confidence, 4),
                "rationale": self.stage2_rationale,
            } if self.routed_to_stage2 else None,
            "final": {
                "finding": self.final_finding,
                "confidence": round(self.final_confidence, 4),
                "severity": self.final_severity,
                "explanation": self.explanation,
                "secondary_findings": self.secondary_findings,
            },
            "flags": {
                "needs_radiologist_review": self.needs_radiologist_review,
                "conflict_detected": self.conflict_detected,
                "conflict_detail": self.conflict_detail,
            },
        }


class ReconciliationLayer:
    """
    Fuses Stage 1 (CNN probability) and Stage 2 (VLM verdict) into
    a single reconciled output.

    Logic:
      1. If only Stage 1 ran (high confidence, skipped Stage 2):
         → Final = Stage 1 result directly
      2. If both stages ran and agree:
         → Final = weighted average, high confidence
      3. If both stages ran and disagree:
         → Flag conflict, trigger radiologist review
    """

    def __init__(
        self,
        weight_stage1: float = RECON_WEIGHT_STAGE1,
        weight_stage2: float = RECON_WEIGHT_STAGE2,
        conflict_threshold: float = CONFLICT_THRESHOLD,
    ):
        self.w1 = weight_stage1
        self.w2 = weight_stage2
        self.conflict_threshold = conflict_threshold

    def reconcile(
        self,
        stage1_probs: np.ndarray,     # (14,) calibrated probabilities
        stage2_result: Optional[dict],  # structured VLM output or None
        routed_to_stage2: bool,
        temperature: float = 1.0,
        image_id: str = "",
        image_path: str = "",
    ) -> PipelineResult:
        """Produce a final PipelineResult from both stage outputs."""

        result = PipelineResult(
            image_id=image_id,
            image_path=image_path,
            stage1_probs=stage1_probs,
            temperature=temperature,
            routed_to_stage2=routed_to_stage2,
        )

        # ── Stage 1 summary ──────────────────────────────────────────────
        top_idx = int(np.argmax(stage1_probs))
        result.stage1_top_finding = DISEASE_LABELS[top_idx]
        result.stage1_top_prob = float(stage1_probs[top_idx])
        result.stage1_max_prob = float(np.max(stage1_probs))

        # ── Path A: Stage 2 was NOT invoked (high confidence fast path) ──
        if not routed_to_stage2 or stage2_result is None:
            result.final_finding = result.stage1_top_finding
            result.final_confidence = result.stage1_top_prob
            result.final_severity = self._severity_from_prob(result.stage1_top_prob)
            result.explanation = (
                f"High-confidence Stage 1 classification: "
                f"{result.stage1_top_finding} (p={result.stage1_top_prob:.3f}, "
                f"T={temperature:.3f}). Stage 2 VLM was not required."
            )
            result.secondary_findings = [
                DISEASE_LABELS[i] for i in range(NUM_CLASSES)
                if i != top_idx and stage1_probs[i] > 0.3
            ]
            return result

        # ── Path B: Both stages ran — need reconciliation ────────────────
        result.stage2_result = stage2_result
        result.stage2_finding = stage2_result.get("finding", "Unknown")
        result.stage2_confidence = stage2_result.get("confidence", 0.5)
        result.stage2_rationale = stage2_result.get("rationale", "")

        # Map VLM finding back to disease index
        s2_finding_idx = self._find_label_index(result.stage2_finding)

        # Check agreement
        stages_agree = (
            s2_finding_idx == top_idx or
            result.stage2_finding.lower() == result.stage1_top_finding.lower()
        )

        if stages_agree:
            # ── Agreement → weighted fusion ──────────────────────────────
            fused_confidence = (
                self.w1 * result.stage1_top_prob +
                self.w2 * result.stage2_confidence
            )
            result.final_finding = result.stage1_top_finding
            result.final_confidence = fused_confidence
            result.final_severity = stage2_result.get(
                "severity", self._severity_from_prob(fused_confidence)
            )
            result.explanation = (
                f"Both stages agree: {result.final_finding}. "
                f"Stage 1 p={result.stage1_top_prob:.3f}, "
                f"Stage 2 confidence={result.stage2_confidence:.3f}. "
                f"Fused confidence={fused_confidence:.3f}. "
                f"VLM rationale: {result.stage2_rationale}"
            )

        else:
            # ── Disagreement → conflict detection ────────────────────────
            confidence_gap = abs(
                result.stage1_top_prob - result.stage2_confidence
            )

            result.conflict_detected = True
            result.conflict_detail = (
                f"Stage 1 says '{result.stage1_top_finding}' "
                f"(p={result.stage1_top_prob:.3f}), "
                f"Stage 2 says '{result.stage2_finding}' "
                f"(confidence={result.stage2_confidence:.3f}). "
                f"Gap={confidence_gap:.3f}."
            )

            # In disagreement, lean toward the higher-confidence stage
            if result.stage2_confidence > result.stage1_top_prob:
                result.final_finding = result.stage2_finding
                result.final_confidence = result.stage2_confidence
            else:
                result.final_finding = result.stage1_top_finding
                result.final_confidence = result.stage1_top_prob

            result.final_severity = self._severity_from_prob(
                result.final_confidence
            )
            result.needs_radiologist_review = True
            result.explanation = (
                f"CONFLICT: {result.conflict_detail} "
                f"Flagged for radiologist review. "
                f"VLM rationale: {result.stage2_rationale}"
            )

        result.secondary_findings = list(set(
            [DISEASE_LABELS[i] for i in range(NUM_CLASSES)
             if i != top_idx and stage1_probs[i] > 0.3] +
            stage2_result.get("secondary_findings", [])
        ))

        return result

    def _find_label_index(self, finding: str) -> int:
        """Map a finding string to DISEASE_LABELS index. -1 if not found."""
        finding_lower = finding.lower().strip()
        for i, label in enumerate(DISEASE_LABELS):
            if label.lower() == finding_lower:
                return i
        # Fuzzy match: check if finding contains a label name
        for i, label in enumerate(DISEASE_LABELS):
            if label.lower() in finding_lower or finding_lower in label.lower():
                return i
        return -1

    @staticmethod
    def _severity_from_prob(prob: float) -> str:
        """Heuristic severity from probability."""
        if prob < 0.3:
            return "normal"
        elif prob < 0.5:
            return "mild"
        elif prob < 0.7:
            return "moderate"
        else:
            return "severe"


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    recon = ReconciliationLayer()

    # Test 1: Stage 1 only (high confidence)
    probs = np.random.rand(14)
    probs[3] = 0.97  # Infiltration is very confident
    r1 = recon.reconcile(probs, None, routed_to_stage2=False,
                         image_id="test_001.png")
    print("=== Stage 1 Only ===")
    print(json.dumps(r1.to_dict(), indent=2, default=str))

    # Test 2: Both stages agree
    s2 = {"finding": "Infiltration", "severity": "moderate",
           "confidence": 0.85, "rationale": "Bilateral opacities",
           "secondary_findings": ["Effusion"]}
    r2 = recon.reconcile(probs, s2, routed_to_stage2=True,
                         image_id="test_002.png")
    print("\n=== Agreement ===")
    print(json.dumps(r2.to_dict(), indent=2, default=str))

    # Test 3: Stages disagree
    s3 = {"finding": "Pneumonia", "severity": "severe",
           "confidence": 0.9, "rationale": "Lobar consolidation",
           "secondary_findings": []}
    r3 = recon.reconcile(probs, s3, routed_to_stage2=True,
                         image_id="test_003.png")
    print("\n=== Conflict ===")
    print(json.dumps(r3.to_dict(), indent=2, default=str))
