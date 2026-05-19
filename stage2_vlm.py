"""
Stage 2 — VLM second opinion using CheXagent (or a fallback rule-based system).

When the real CheXagent model is available, uncertain cases are run through
the VLM with multiple prompt formulations, and a majority vote is taken.

Output is forced into structured JSON:
  {finding, severity, confidence, rationale}

Fallback: if the VLM cannot be loaded (no GPU memory, missing deps),
a rule-based heuristic generates a structured response from Stage 1 logits.
"""

import json
import re
from typing import Optional

import numpy as np
import torch
from PIL import Image

from config import (
    DISEASE_LABELS, NUM_CLASSES,
    VLM_MODEL_ID, VLM_NUM_PROMPTS, VLM_MAX_NEW_TOKENS,
    get_device,
)


# ══════════════════════════════════════════════════════════════════════════════
# Structured output schema
# ══════════════════════════════════════════════════════════════════════════════
STRUCTURED_SCHEMA = {
    "finding": "str — primary finding name",
    "severity": "str — one of: normal, mild, moderate, severe",
    "confidence": "float — 0.0 to 1.0",
    "rationale": "str — brief clinical reasoning",
    "secondary_findings": "list[str] — other findings if any",
}


# ══════════════════════════════════════════════════════════════════════════════
# Prompt formulations for majority voting
# ══════════════════════════════════════════════════════════════════════════════
PROMPT_TEMPLATES = [
    # Prompt 1 — Direct diagnosis
    """Analyze this chest X-ray image carefully. Identify any abnormalities.

Respond ONLY with valid JSON in this exact format:
{{
    "finding": "<primary finding or 'Normal'>",
    "severity": "<normal|mild|moderate|severe>",
    "confidence": <0.0 to 1.0>,
    "rationale": "<brief clinical reasoning>",
    "secondary_findings": ["<other finding 1>", "<other finding 2>"]
}}

Focus on these 14 conditions: {labels}.
If the image appears normal, set finding to "Normal" and severity to "normal".""",

    # Prompt 2 — Systematic review
    """You are an expert radiologist reviewing a chest X-ray.
Perform a systematic review: check lung fields, heart silhouette,
mediastinum, costophrenic angles, and bony thorax.

The possible conditions are: {labels}.

Output your assessment as JSON only (no other text):
{{
    "finding": "<primary abnormality or 'Normal'>",
    "severity": "<normal|mild|moderate|severe>",
    "confidence": <confidence score 0-1>,
    "rationale": "<systematic reasoning>",
    "secondary_findings": ["<any additional findings>"]
}}""",
]


# ══════════════════════════════════════════════════════════════════════════════
# VLM wrapper
# ══════════════════════════════════════════════════════════════════════════════
class Stage2VLM:
    """
    Wraps CheXagent (or compatible VLM) for chest X-ray analysis.

    Runs multiple prompt formulations and takes majority vote for
    consistency — catches prompt-sensitive hallucinations.
    """

    def __init__(self, model_id: str = VLM_MODEL_ID, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device or str(get_device())
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> bool:
        """
        Attempt to load the VLM. Returns True if successful.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            print(f"[Stage2] Loading VLM: {self.model_id} ...")

            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.model.eval()
            self._loaded = True
            print(f"[Stage2] VLM loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"[Stage2] WARNING: Failed to load VLM: {e}")
            print("[Stage2] Will use rule-based fallback for Stage 2.")
            self._loaded = False
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _run_single_prompt(
        self,
        image: Image.Image,
        prompt_template: str,
    ) -> dict:
        """Run one prompt through the VLM and parse structured JSON output."""
        prompt = prompt_template.format(labels=", ".join(DISEASE_LABELS))

        try:
            # Process inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=VLM_MAX_NEW_TOKENS,
                    do_sample=False,         # deterministic for consistency
                    temperature=1.0,
                )

            # Decode
            generated = self.processor.decode(
                outputs[0], skip_special_tokens=True
            )

            # Extract JSON from response
            return self._parse_json_response(generated)

        except Exception as e:
            print(f"[Stage2] VLM inference error: {e}")
            return self._default_response(f"VLM error: {str(e)}")

    def _parse_json_response(self, text: str) -> dict:
        """Extract and validate JSON from VLM output."""
        # Try to find JSON block in response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                # Validate required fields
                result = {
                    "finding": str(parsed.get("finding", "Unknown")),
                    "severity": str(parsed.get("severity", "unknown")),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "rationale": str(parsed.get("rationale", "No rationale provided")),
                    "secondary_findings": list(parsed.get("secondary_findings", [])),
                }
                # Clamp confidence
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
                return result
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        return self._default_response(f"Could not parse VLM output: {text[:200]}")

    def _default_response(self, rationale: str = "Fallback") -> dict:
        return {
            "finding": "Unknown",
            "severity": "unknown",
            "confidence": 0.5,
            "rationale": rationale,
            "secondary_findings": [],
        }

    def analyze(
        self,
        image: Image.Image,
        num_prompts: int = VLM_NUM_PROMPTS,
    ) -> dict:
        """
        Run the image through multiple prompt formulations and take
        majority vote on the primary finding.

        Returns a single structured dict with the consensus result
        plus a `votes` field showing individual prompt results.
        """
        if not self._loaded:
            return self._default_response("VLM not loaded")

        prompts = PROMPT_TEMPLATES[:num_prompts]
        votes = []

        for i, template in enumerate(prompts):
            result = self._run_single_prompt(image, template)
            result["prompt_id"] = i
            votes.append(result)

        # Majority vote on primary finding
        findings = [v["finding"] for v in votes]
        from collections import Counter
        finding_counts = Counter(findings)
        consensus_finding = finding_counts.most_common(1)[0][0]

        # Average confidence across votes with matching finding
        matching = [v for v in votes if v["finding"] == consensus_finding]
        avg_confidence = np.mean([v["confidence"] for v in matching])

        # Use rationale from the first matching vote
        primary_rationale = matching[0]["rationale"] if matching else "No consensus"

        # Collect all secondary findings
        all_secondary = set()
        for v in votes:
            all_secondary.update(v.get("secondary_findings", []))

        consensus = {
            "finding": consensus_finding,
            "severity": matching[0]["severity"] if matching else "unknown",
            "confidence": float(avg_confidence),
            "rationale": primary_rationale,
            "secondary_findings": list(all_secondary),
            "votes": votes,
            "consensus_strength": len(matching) / len(votes),
        }

        return consensus


# ══════════════════════════════════════════════════════════════════════════════
# Rule-based fallback (when VLM is unavailable)
# ══════════════════════════════════════════════════════════════════════════════
class Stage2Fallback:
    """
    Rule-based fallback that uses Stage 1's logits to generate
    a structured response mimicking VLM output.

    This is used when CheXagent cannot be loaded (e.g., insufficient
    GPU memory, missing dependencies).
    """

    def analyze_from_logits(
        self,
        logits: np.ndarray,           # (14,) raw logits from Stage 1
        probs: np.ndarray,            # (14,) sigmoid probabilities
    ) -> dict:
        """Generate structured output from Stage 1 predictions."""
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_finding = DISEASE_LABELS[top_idx]

        # Determine severity from probability
        if top_prob < 0.3:
            severity = "normal"
            finding = "Normal"
        elif top_prob < 0.5:
            severity = "mild"
            finding = top_finding
        elif top_prob < 0.7:
            severity = "moderate"
            finding = top_finding
        else:
            severity = "severe"
            finding = top_finding

        # Secondary findings: anything above 0.3 that isn't the primary
        secondary = [
            DISEASE_LABELS[i] for i in range(NUM_CLASSES)
            if i != top_idx and probs[i] > 0.3
        ]

        # Build rationale
        rationale_parts = [f"Primary finding: {finding} (p={top_prob:.2f})."]
        if secondary:
            rationale_parts.append(
                f"Secondary findings above threshold: {', '.join(secondary)}."
            )
        rationale_parts.append(
            "Based on EfficientNet-B4 logit analysis (rule-based fallback, "
            "VLM unavailable)."
        )

        return {
            "finding": finding,
            "severity": severity,
            "confidence": top_prob,
            "rationale": " ".join(rationale_parts),
            "secondary_findings": secondary,
            "votes": [],  # no VLM votes
            "consensus_strength": 1.0,  # single source
            "is_fallback": True,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Unified Stage 2 interface
# ══════════════════════════════════════════════════════════════════════════════
class Stage2Engine:
    """
    Unified interface for Stage 2 analysis.
    Tries VLM first; falls back to rule-based if VLM unavailable.
    """

    def __init__(self, try_vlm: bool = True, model_id: str = VLM_MODEL_ID):
        self.vlm = None
        self.fallback = Stage2Fallback()
        self.using_vlm = False

        if try_vlm:
            self.vlm = Stage2VLM(model_id=model_id)
            self.using_vlm = self.vlm.load()

        if not self.using_vlm:
            print("[Stage2Engine] Using rule-based fallback")

    def analyze(
        self,
        image: Optional[Image.Image] = None,
        logits: Optional[np.ndarray] = None,
        probs: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Analyze a case using the best available method.

        Args:
            image: PIL image (required for VLM)
            logits: Stage 1 raw logits (used for fallback)
            probs: Stage 1 probabilities (used for fallback)
        """
        if self.using_vlm and image is not None:
            return self.vlm.analyze(image)
        elif logits is not None and probs is not None:
            return self.fallback.analyze_from_logits(logits, probs)
        else:
            return self.fallback.analyze_from_logits(
                np.zeros(NUM_CLASSES), np.ones(NUM_CLASSES) * 0.5
            )


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test fallback
    print("Testing Stage2Fallback ...")
    fb = Stage2Fallback()
    dummy_probs = np.random.rand(14)
    dummy_logits = np.random.randn(14)
    result = fb.analyze_from_logits(dummy_logits, dummy_probs)
    print(json.dumps(result, indent=2))

    # Test engine (will likely use fallback without GPU)
    print("\nTesting Stage2Engine ...")
    engine = Stage2Engine(try_vlm=False)
    result = engine.analyze(logits=dummy_logits, probs=dummy_probs)
    print(json.dumps(result, indent=2))
