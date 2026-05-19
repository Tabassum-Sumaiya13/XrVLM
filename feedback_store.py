"""
Feedback store — persists radiologist corrections for continuous learning.

When a radiologist reviews and corrects a case, the correction is stored
in a JSON-lines file. This closes the feedback loop:

  correction → stored → used to retune temperature scaling

The store also tracks statistics (agreement rate, correction rate, etc.)
for monitoring pipeline health.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from config import FEEDBACK_DB_PATH, DISEASE_LABELS


@dataclass
class CorrectionRecord:
    """A single radiologist correction."""
    timestamp: float                     # Unix epoch
    image_id: str                        # filename
    pipeline_finding: str                # what the pipeline said
    pipeline_confidence: float           # pipeline's confidence
    radiologist_finding: str             # what the radiologist corrected to
    radiologist_severity: str            # radiologist-assessed severity
    was_correct: bool                    # did pipeline match radiologist?
    was_routed_to_stage2: bool           # did this case go through VLM?
    conflict_detected: bool              # was a conflict flagged?
    notes: str = ""                      # free-text radiologist notes

    # For temperature recalibration
    stage1_logits: Optional[list] = None    # (14,) raw logits
    stage1_probs: Optional[list] = None     # (14,) calibrated probs
    true_labels: Optional[list] = None      # (14,) ground truth binary


class FeedbackStore:
    """
    JSON-lines based feedback store.

    Each line = one CorrectionRecord serialised as JSON.
    Append-only for safety; reads load all records.
    """

    def __init__(self, path: Path = FEEDBACK_DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")  # create empty file

    def add_correction(self, record: CorrectionRecord) -> None:
        """Append a correction to the store."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), default=str) + "\n")

    def add_from_dict(
        self,
        image_id: str,
        pipeline_finding: str,
        pipeline_confidence: float,
        radiologist_finding: str,
        radiologist_severity: str = "unknown",
        was_routed_to_stage2: bool = False,
        conflict_detected: bool = False,
        notes: str = "",
        stage1_logits: Optional[list] = None,
        stage1_probs: Optional[list] = None,
        true_labels: Optional[list] = None,
    ) -> CorrectionRecord:
        """Convenience: build and store a CorrectionRecord from kwargs."""
        was_correct = (
            pipeline_finding.lower().strip() ==
            radiologist_finding.lower().strip()
        )

        record = CorrectionRecord(
            timestamp=time.time(),
            image_id=image_id,
            pipeline_finding=pipeline_finding,
            pipeline_confidence=pipeline_confidence,
            radiologist_finding=radiologist_finding,
            radiologist_severity=radiologist_severity,
            was_correct=was_correct,
            was_routed_to_stage2=was_routed_to_stage2,
            conflict_detected=conflict_detected,
            notes=notes,
            stage1_logits=stage1_logits,
            stage1_probs=stage1_probs,
            true_labels=true_labels,
        )
        self.add_correction(record)
        return record

    def load_all(self) -> list[CorrectionRecord]:
        """Read all corrections from the store."""
        records = []
        if not self.path.exists():
            return records

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(CorrectionRecord(**d))
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"[FeedbackStore] Skipping malformed record: {e}")
        return records

    def count(self) -> int:
        """Number of corrections in the store."""
        if not self.path.exists():
            return 0
        with open(self.path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_statistics(self) -> dict:
        """Compute summary statistics over all corrections."""
        records = self.load_all()
        if not records:
            return {
                "total_corrections": 0,
                "accuracy": 0.0,
                "stage2_usage_rate": 0.0,
                "conflict_rate": 0.0,
            }

        total = len(records)
        correct = sum(1 for r in records if r.was_correct)
        stage2_used = sum(1 for r in records if r.was_routed_to_stage2)
        conflicts = sum(1 for r in records if r.conflict_detected)

        # Per-finding accuracy
        finding_stats = {}
        for r in records:
            gt = r.radiologist_finding
            if gt not in finding_stats:
                finding_stats[gt] = {"total": 0, "correct": 0}
            finding_stats[gt]["total"] += 1
            if r.was_correct:
                finding_stats[gt]["correct"] += 1

        return {
            "total_corrections": total,
            "accuracy": correct / total if total > 0 else 0.0,
            "stage2_usage_rate": stage2_used / total if total > 0 else 0.0,
            "conflict_rate": conflicts / total if total > 0 else 0.0,
            "per_finding": finding_stats,
        }

    def get_calibration_data(self) -> tuple:
        """
        Extract (logits, true_labels) pairs for temperature recalibration.
        Returns (logits_array, labels_array) as numpy arrays.
        Only includes records with both logits and true_labels present.
        """
        import numpy as np

        records = self.load_all()
        logits_list = []
        labels_list = []

        for r in records:
            if r.stage1_logits is not None and r.true_labels is not None:
                logits_list.append(r.stage1_logits)
                labels_list.append(r.true_labels)

        if not logits_list:
            return None, None

        return np.array(logits_list), np.array(labels_list)

    def clear(self) -> None:
        """Clear all corrections (use with caution)."""
        self.path.write_text("")


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))

    store = FeedbackStore(path=tmp)
    print(f"Store path: {tmp}")
    print(f"Initial count: {store.count()}")

    # Add some test corrections
    store.add_from_dict(
        image_id="test_001.png",
        pipeline_finding="Infiltration",
        pipeline_confidence=0.85,
        radiologist_finding="Infiltration",
        radiologist_severity="moderate",
        notes="Correct identification",
        stage1_logits=[0.1] * 14,
        stage1_probs=[0.5] * 14,
        true_labels=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )

    store.add_from_dict(
        image_id="test_002.png",
        pipeline_finding="Pneumonia",
        pipeline_confidence=0.72,
        radiologist_finding="Consolidation",
        radiologist_severity="severe",
        was_routed_to_stage2=True,
        conflict_detected=True,
        notes="Pipeline misidentified consolidation as pneumonia",
    )

    print(f"After adds: {store.count()}")
    stats = store.get_statistics()
    print(f"Stats: {json.dumps(stats, indent=2)}")

    # Cleanup
    tmp.unlink()
