# Two-Stage CXR Diagnostic Pipeline

A production-grade, two-stage chest X-ray diagnostic system that combines fast CNN screening with VLM-powered second opinions, calibrated confidence scoring, and continuous learning from radiologist feedback.

> **Disclaimer:** This project is for research and educational purposes only.  
> It is **not** a medical device and must **not** be used for clinical decision-making.

---

## Architecture

```
                        ┌──────────────────┐
                        │  ① Raw CXR Image │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                        │ ② EfficientNet-B4 │
                        │  + Temperature    │
                        │    Scaling        │
                        └────────┬─────────┘
                                 │
                        ┌────────▼─────────┐
                   ┌────┤ ③ p ≥ τ (0.95)?  ├────┐
                   │YES └──────────────────┘ NO │
                   │                            │
           ┌───────▼──────┐            ┌────────▼────────┐
           │  Fast Accept  │            │ ④ CheXagent VLM │
           │  (skip VLM)   │            │  (2× prompts +  │
           │               │            │   majority vote) │
           └───────┬──────┘            └────────┬────────┘
                   │                            │
                   └──────────┬─────────────────┘
                              │
                     ┌────────▼─────────┐
                     │ ⑤ Reconciliation  │
                     │  (weighted fusion │
                     │  + conflict flag) │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │ ⑥ Final Decision  │
                     │  + Explanation    │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │ ⑦ Feedback Store  │◄── Radiologist
                     └────────┬─────────┘    corrections
                              │
                     ┌────────▼─────────┐
                     │ ⑧ Calibration    │
                     │    Update (T)    │
                     └──────────────────┘
```

## Key Features

| Feature | Description |
|---|---|
| **Stage 1 — Fast Screening** | EfficientNet-B4 with temperature-scaled calibrated probabilities |
| **Temperature Scaling** | Post-hoc calibration (single learned T parameter) corrects overconfidence |
| **Routing Decision** | High-confidence cases skip the expensive VLM (efficiency gain) |
| **Stage 2 — VLM** | CheXagent with multi-prompt majority voting for consistency |
| **Reconciliation** | Weighted fusion of both stages; conflicts trigger radiologist review |
| **Feedback Loop** | Radiologist corrections stored and used to retune calibration |
| **Structured Output** | All results in parseable JSON (finding, severity, confidence, rationale) |

## Quick Start

```bash
# 1. Clone and setup
cd chest-xray-pipeline
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Generate sample data + run full pipeline demo
python run_sample.py --n-images 100

# 3. Or run each step individually:
python download_sample.py --n 100       # generate sample data
python train.py --sample 100            # train on sample
python pipeline.py path/to/xray.png     # analyse single image
```

## Project Structure

```
chest-xray-pipeline/
├── config.py              # Central configuration (paths, thresholds, labels)
├── dataset.py             # NIH CXR-14 dataset and DataLoader setup
├── stage1_model.py        # EfficientNet-B4 + temperature scaling
├── stage2_vlm.py          # CheXagent VLM + fallback rule-based system
├── reconciliation.py      # Weighted fusion + conflict detection
├── feedback_store.py      # Radiologist correction persistence
├── train.py               # Two-stage training loop
├── pipeline.py            # Full pipeline orchestrator
├── run_sample.py          # End-to-end sample demo
├── download_sample.py     # Sample data generation
├── requirements.txt       # Dependencies
│
├── data/
│   └── raw/               # NIH CXR-14 images + metadata
├── outputs/
│   ├── checkpoints/       # Model checkpoints
│   ├── figures/           # Visualisations
│   └── feedback/          # Radiologist corrections
```

## Pipeline Stages Explained

### ② Stage 1 — EfficientNet-B4 Fast Screening
- **Backbone:** EfficientNet-B4 pretrained on ImageNet, fine-tuned on CXR-14
- **Head:** `Dropout(0.4) → Linear(1792 → 14)`
- **Output:** 14 independent sigmoid probabilities (multi-label)
- **Temperature scaling:** Learned parameter T divides logits before sigmoid

### ③ Routing Decision
- Calibrated `max(prob)` compared to threshold τ (default 0.95)
- High confidence → accept directly (skip Stage 2)
- Low confidence → route to Stage 2 for VLM analysis

### ④ Stage 2 — VLM Second Opinion
- CheXagent vision-language model (or rule-based fallback)
- 2 different prompt formulations → majority vote
- Structured JSON output: `{finding, severity, confidence, rationale}`

### ⑤ Reconciliation Layer
- **Agreement:** Weighted average → `0.4 × Stage1 + 0.6 × Stage2`
- **Disagreement:** Flag conflict → trigger radiologist review

### ⑦–⑧ Feedback + Calibration
- Radiologist corrections stored in JSON-lines format
- Every N corrections, temperature T is retuned via grid search
- Continuous improvement without full model retraining

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 380 | EfficientNet-B4 native input size |
| `ROUTING_THRESHOLD` | 0.95 | τ — confidence threshold for routing |
| `TEMPERATURE_INIT` | 1.5 | Initial temperature for calibration |
| `RECON_WEIGHT_STAGE1` | 0.4 | Stage 1 weight in fusion |
| `RECON_WEIGHT_STAGE2` | 0.6 | Stage 2 weight in fusion |
| `CONFLICT_THRESHOLD` | 0.3 | Disagreement gap to flag conflict |

## Dataset

**NIH Chest X-Ray 14** — Wang et al. 2017
- 112,120 frontal-view X-ray images, 14 disease labels
- For sample runs, synthetic images are generated automatically

## References

- Wang et al. (2017). *ChestX-ray8: Hospital-scale Chest X-ray Database*. CVPR.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs*. ICML.
- Guo et al. (2017). *On Calibration of Modern Neural Networks*. ICML.
- Chen et al. (2024). *CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation*. 

## License

MIT — see LICENSE for details.
