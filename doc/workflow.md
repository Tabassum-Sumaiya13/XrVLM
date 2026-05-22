
# xrVLM: Chest X-ray Diagnostic Pipeline with VLM Integration

## Overview

xrVLM is a two-stage diagnostic pipeline for chest X-ray analysis that combines a calibrated CNN (EfficientNet-B4) with a Vision-Language Model (CheXagent) through intelligent routing and reconciliation. The system achieves high efficiency by using the fast CNN for confident cases and only invoking the slower VLM for uncertain ones, while maintaining interpretability through natural language explanations.

### Key Features
- **Two-stage architecture**: Fast CNN for routine cases, VLM for complex ones
- **Adaptive routing**: Routes uncertain cases (max confidence < τ) to VLM
- **Temperature scaling**: Calibrated probability estimates
- **Structured reconciliation**: Weighted fusion with conflict detection
- **Feedback loop**: Continuous improvement from radiologist corrections



## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Input X-ray                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: CNN (EfficientNet-B4)               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Preprocess (384×384, normalize)                      │    │
│  │  • Extract features                                     │    │
│  │  • Classify 14 pathologies                              │    │
│  │  • Temperature scaling for calibration                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        Output: Probs[14] + max confidence       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ max(probs) ≥ τ? │
                    └─────────────────┘
                     │              │
                 ┌──┴──┐        ┌──┴──┐
                 │ YES │        │ NO  │
                 └──┬──┘        └──┬──┘
                    │              │
                    ▼              ▼
           ┌────────────┐   ┌─────────────────────────┐
           │ Use Stage 1│   │ Stage 2: VLM (CheXagent)│
           │   only     │   │  • Dual prompt templates│
           └────────────┘   │  • Majority voting (3x) │
                            │  • JSON parsing         │
                            │  • Rule-based fallback  │
                            └─────────────────────────┘
                    │              │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │   Reconciliation       │
              │  • Weighted fusion     │
              │  • Conflict detection  │
              │  • Explanation gen     │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Final Diagnosis      │
              │  + Explanation         │
              │  + Confidence score    │
              └────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   Feedback Store       │
              │  (Radiologist review)  │
              └────────────────────────┘

```

---

## File-by-File Breakdown

### Configuration (`config.py`)
**Purpose**: Central configuration hub for the entire pipeline.

| Component | Settings |
|-----------|----------|
| Paths | `DATA_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`, `FEEDBACK_PATH` |
| Model | Backbone (EfficientNet-B4), 14 classes, dropout 0.2 |
| Training | Batch size 32, LR 1e-4, warmup 10 epochs, finetune 20 epochs |
| Pipeline | τ=0.7 routing threshold, conflict threshold 0.3 |
| Reconciliation | Stage1 weight 0.4, Stage2 weight 0.6 |

### Data Management (`dataset.py`)
**Purpose**: Load NIH ChestX-ray2017 dataset with proper splits.

```
Input: Data_Entry_2017.csv + image directories
       train_val_list.txt, test_list.txt (optional)

Process:
  1. Parse CSV metadata (Image Index, Finding Labels)
  2. Apply official splits if available
  3. Create train/val/test indices
  4. Apply transforms:
     - Train: RandomHorizontalFlip, RandomRotation, ColorJitter
     - Val/Test: Center crop + normalization
  5. Compute class weights for imbalance

Output: train_loader, val_loader, test_loader, pos_weight tensor
```

**Label Set** (14 pathologies):
```
Atelectasis, Consolidation, Infiltration, Pneumothorax,
Edema, Emphysema, Fibrosis, Effusion, Pneumonia,
Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia
```

### Stage 1 Model (`stage1_model.py`)
**Purpose**: CNN backbone with calibration.

```python
class Stage1Model(nn.Module):
    - EfficientNet-B4 backbone
    - Classifier head (1280 → 512 → 14)
    - Dropout 0.2 for regularization

class TemperatureScaler:
    - Single parameter T > 0
    - Scales logits: logits / T
    - Optimized via NLL loss on validation set
```

**Training Flow**:
1. **Warmup phase** (10 epochs): Backbone frozen, train head only
2. **Finetune phase** (20 epochs): Full network training
3. **Temperature scaling**: Tune T on validation set after training

### Training Loop (`train.py`)
**Purpose**: Orchestrates two-phase training with logging.

```
┌────────────────────────────────────────────────────────────┐
│                     Training Pipeline                       │
├────────────────────────────────────────────────────────────┤
│  Phase 1: Warmup                                           │
│  ├── Freeze backbone                                       │
│  ├── Train head for 10 epochs                              │
│  └── Save best model (val loss)                           │
├────────────────────────────────────────────────────────────┤
│  Phase 2: Finetune                                         │
│  ├── Unfreeze backbone                                     │
│  ├── Train full model for 20 epochs                        │
│  ├── Learning rate decay (step 10, gamma 0.1)             │
│  └── Save best model (val loss)                           │
├────────────────────────────────────────────────────────────┤
│  Post-processing                                           │
│  ├── Temperature scaling on validation set                 │
│  └── Save calibrated checkpoint                           │
└────────────────────────────────────────────────────────────┘

Outputs:
  - checkpoints/best_model.pth
  - checkpoints/best_model_calibrated.pth
  - logs/loss_log.csv (epoch, train_loss, val_loss, lr)
```

### Stage 2 VLM (`stage2_vlm.py`)
**Purpose**: Vision-Language Model integration with fallback.

```python
class Stage2Engine:
    def __init__(self):
        # Primary: Load CheXagent from Hugging Face
        # Fallback: Rule-based system using Stage 1 outputs
    
    def predict(image_path, stage1_probs):
        """
        Process:
          1. Prepare two prompt templates:
             - Template A: Direct finding identification
             - Template B: Comparative analysis
          2. Run VLM 3 times with different temperatures
          3. Majority voting on JSON outputs
          4. Parse structured response:
             {
               "primary_finding": str,
               "confidence": float,
               "supporting_evidence": str,
               "differential": list
             }
        """
```

**Fallback Logic** (when VLM unavailable):
- Use Stage 1 probabilities
- Apply rule-based thresholding (p > 0.5)
- Generate template-based explanations

### Pipeline (`pipeline.py`)
**Purpose**: End-to-end inference orchestration.

```python
class DiagnosticPipeline:
    def predict(image_path):
        # Step 1: Load and preprocess
        image = preprocess(image_path)  # 384×384, normalize
        
        # Step 2: Stage 1 inference
        logits = stage1_model(image)
        probs = torch.sigmoid(logits)
        max_conf = max(probs)
        
        # Step 3: Routing decision
        if max_conf >= τ:
            # Confident case - skip Stage 2
            final = stage1_only_reconcile(probs)
        else:
            # Uncertain case - invoke Stage 2
            vlm_output = stage2_engine.predict(image_path, probs)
            final = reconcile(probs, vlm_output)
        
        # Step 4: Return structured result
        return PipelineResult(
            findings=final.finding,
            explanation=final.explanation,
            confidence=final.confidence,
            conflict=final.conflict_detected
        )
```

### Reconciliation (`reconciliation.py`)
**Purpose**: Fuse Stage 1 and Stage 2 predictions.

**Algorithm**:
```python
def reconcile_predictions(stage1_probs, stage2_output, stage1_conf):
    if stage2_output is None:
        return stage1_only_result
    
    # Agreement check
    stage1_finding = argmax(stage1_probs)
    stage2_finding = stage2_output['primary_finding']
    
    if stage1_finding == stage2_finding:
        # Weighted confidence fusion
        final_conf = (w1 * stage1_conf + 
                      w2 * stage2_output['confidence'])
        return final_result
    else:
        # Conflict detected
        if disagreement_magnitude > conflict_threshold:
            return conflict_result(requires_review=True)
        else:
            return weighted_result_with_uncertainty
```

**Conflict Handling**:
- Disagreement > 0.3 → Flag for radiologist review
- Store conflict cases separately for analysis

### Feedback Store (`feedback_store.py`)
**Purpose**: Persistent storage of radiologist corrections.

```python
class FeedbackStore:
    corrections.json format:
    {
        "timestamp": "2024-01-15T10:30:00",
        "image_path": "path/to/xray.png",
        "original_prediction": {...},
        "corrected_labels": {"Cardiomegaly": true, ...},
        "radiologist_id": "dr_smith",
        "confidence": 0.95
    }
    
    Methods:
    - compute_accuracy(): Overall and per-label accuracy
    - get_calibration_data(): Extract (logits, labels) pairs
    - get_conflict_rate(): Proportion of clinician-VLM disagreements
    - export_report(): Generate JSON analytics report
```

### Demo Scripts

#### `run_sample.py` - Full End-to-End Demo
```
Step 1: Data Acquisition
  └── Download NIH ChestX-ray2017 via Kaggle API

Step 2: Training
  └── Train Stage 1 CNN (2-phase + calibration)

Step 3: Inference
  ├── Load calibrated model
  ├── Run pipeline on 100 test images
  └── Track routing decisions and conflicts

Step 4: Simulated Feedback
  ├── Generate mock radiologist corrections
  ├── Store in feedback store
  └── Compute accuracy metrics

Output: sample_results.json
```

#### `download_sample.py` - Lightweight Sample Download
```
Purpose: Quick testing without full Kaggle download

Process:
  1. Stream 500 samples from Hugging Face datasets
  2. Generate minimal Data_Entry_2017.csv
  3. Create split files for train/val/test
  4. Download corresponding images

Use case: Development and testing only
```

---

## Data Flow Summary

### Training Phase
```
Raw Data → dataset.py → DataLoaders → train.py → Stage1Model → TemperatureScaler → calibrated_model.pth
                                              ↓
                                         loss_log.csv
```

### Inference Phase
```
Input Image → pipeline.py → Stage1 (CNN)
                               ↓
                        [confidence < τ?]
                          ↓         ↓
                       (no)       (yes)
                          ↓         ↓
                     use Stage1   Stage2 VLM
                          ↓         ↓
                        reconciliation.py
                               ↓
                        final diagnosis
                               ↓
                    feedback_store.py (optional)
```

### Feedback Loop
```
Radiologist Review → corrections.json → compute_accuracy() → calibration_data → retrain/update
```

---

## Configuration Parameters

### Routing Threshold (τ)
- **Range**: 0.5-0.9
- **Default**: 0.7
- **Effect**: Lower τ = fewer VLM calls (faster, less accurate); Higher τ = more VLM calls (slower, potentially more accurate)

### Reconciliation Weights
- **Stage1 weight**: 0.4 (default)
- **Stage2 weight**: 0.6 (default)
- **Effect**: Controls trust in each stage when they agree

### Conflict Threshold
- **Default**: 0.3
- **Effect**: Magnitude of disagreement needed to flag for review

### Training Hyperparameters
| Parameter | Warmup | Finetune |
|-----------|--------|----------|
| Epochs | 10 | 20 |
| Learning rate | 1e-4 | 1e-4 → 1e-5 |
| Backbone | Frozen | Unfrozen |
| Batch size | 32 | 32 |

---

## Output Files

| File | Location | Format | Description |
|------|----------|--------|-------------|
| best_model.pth | checkpoints/ | PyTorch | Best model weights |
| best_model_calibrated.pth | checkpoints/ | PyTorch | Calibrated version |
| loss_log.csv | logs/ | CSV | Training metrics |
| sample_results.json | project root | JSON | Demo inference results |
| corrections.json | project root | JSONL | Radiologist feedback |

---

## Usage Examples

### Basic Inference
```python
from pipeline import DiagnosticPipeline

pipeline = DiagnosticPipeline(
    stage1_model_path="checkpoints/best_model_calibrated.pth",
    tau=0.7
)

result = pipeline.predict("chest_xray.png")
print(f"Finding: {result.final_finding}")
print(f"Confidence: {result.stage1_confidence:.3f}")
print(f"Routed to VLM: {result.routed_to_stage2}")
```

### Training from Scratch
```bash
# Using full dataset
python train.py --data_dir /path/to/nih_data --epochs_warmup 10 --epochs_finetune 20

# Using sample data
python download_sample.py
python train.py --data_dir ./sample_data --quick_test
```

### Running Demo
```bash
# Install dependencies
pip install -r requirements.txt

# Configure Kaggle API (for full data)
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key

# Run demo
python run_sample.py --use_sample_data  # For quick test
python run_sample.py --download_full    # For full pipeline
```

---

## Performance Metrics

Track these metrics during operation:

1. **Routing Rate**: % of cases sent to Stage 2 = 1 - (% confident cases)
2. **Conflict Rate**: % where Stage1 and Stage2 disagree significantly
3. **Calibration Error**: Expected Calibration Error (ECE) after temperature scaling
4. **Clinical Accuracy**: Agreement with radiologist corrections

---

## Limitations & Future Work

**Current Limitations**:
- VLM (CheXagent) requires GPU with >12GB VRAM
- Fallback rule-based system has lower accuracy
- Limited to 14 NIH pathologies
- No temporal or multi-view reasoning

**Planned Improvements**:
- [ ] Active learning for uncertain cases
- [ ] Online calibration updates from feedback
- [ ] Support for DICOM format
- [ ] Integration with PACS systems
- [ ] Explainable AI (Grad-CAM overlays)

---

## License & Citation

[Add your license and citation information here]
```

This improved documentation includes:

1. **Clear architecture diagram** showing data flow
2. **File-by-file breakdown** with purposes and interfaces
3. **Algorithm details** for key components
4. **Configuration parameter explanations** with ranges and effects
5. **Usage examples** for common workflows
6. **Performance metrics** to track
7. **Limitations** section for honest assessment
8. **Better formatting** with tables, code blocks, and ASCII diagrams

The documentation is now more scannable, actionable, and suitable for both developers and clinical collaborators.