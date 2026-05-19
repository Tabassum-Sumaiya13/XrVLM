"""
Central configuration for the two-stage CXR diagnostic pipeline.

All paths, hyper-parameters, thresholds, and label definitions live here
so every module imports from one canonical source.
"""

from pathlib import Path

# ── project layout ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
SAMPLE_DIR   = DATA_DIR / "sample"          # small subset for fast iteration

OUTPUT_DIR   = PROJECT_ROOT / "outputs"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
FIGURES_DIR  = OUTPUT_DIR / "figures"
FEEDBACK_DIR = OUTPUT_DIR / "feedback"

for _d in [DATA_DIR, RAW_DIR, SAMPLE_DIR, OUTPUT_DIR, CKPT_DIR,
           FIGURES_DIR, FEEDBACK_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── NIH CXR-14 label definitions ────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]
NUM_CLASSES = len(DISEASE_LABELS)  # 14

# ── image pre-processing ────────────────────────────────────────────────────
IMAGE_SIZE = 380            # EfficientNet-B4 native input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Stage 1 — EfficientNet-B4 ──────────────────────────────────────────────
STAGE1_BACKBONE    = "efficientnet_b4"
STAGE1_DROPOUT     = 0.4
STAGE1_LR_WARMUP   = 1e-3       # head-only warm-up
STAGE1_LR_FINETUNE = 1e-4       # full fine-tune
STAGE1_EPOCHS_WARMUP   = 3
STAGE1_EPOCHS_FINETUNE = 12
BATCH_SIZE = 16                 # smaller for B4 at 380×380

# ── Temperature scaling ─────────────────────────────────────────────────────
TEMPERATURE_INIT = 1.5          # starting T (>1 = soften, <1 = sharpen)

# ── Routing threshold ───────────────────────────────────────────────────────
ROUTING_THRESHOLD = 0.95        # τ — calibrated p ≥ τ  → skip Stage 2

# ── Stage 2 — VLM (CheXagent) ──────────────────────────────────────────────
VLM_MODEL_ID  = "StanfordAIMI/CheXagent-2-3b"   # HF model ID
VLM_NUM_PROMPTS = 2             # number of prompt formulations for majority vote
VLM_MAX_NEW_TOKENS = 512

# ── Reconciliation ──────────────────────────────────────────────────────────
RECON_WEIGHT_STAGE1 = 0.4       # weight for Stage 1 prob in fusion
RECON_WEIGHT_STAGE2 = 0.6       # weight for Stage 2 verdict in fusion
CONFLICT_THRESHOLD  = 0.3       # |p1 − p2| above this → flag for review

# ── Feedback store ──────────────────────────────────────────────────────────
FEEDBACK_DB_PATH = FEEDBACK_DIR / "corrections.json"
CALIBRATION_UPDATE_INTERVAL = 50  # retune T every N corrections

# ── Device ──────────────────────────────────────────────────────────────────
import torch

def get_device() -> torch.device:
    """CUDA → MPS → CPU priority."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
