# Methodology

## Problem formulation
We study automated chest X-ray (CXR) interpretation as a **multi-label classification** problem over a fixed label set of $C=14$ thoracic findings (NIH CXR-14; Wang et al., 2017). Given an input image $x$, the system produces:

1. A calibrated probability vector $p \in [0,1]^C$ from a fast convolutional model (Stage 1).
2. A structured, natural-language second opinion from a vision–language model (VLM) (Stage 2), invoked only when Stage 1 is uncertain.
3. A reconciled final decision with an explanation and a review flag when the two stages conflict.

The overall design goal is to reduce the cost of VLM inference while preserving reliability by routing only uncertain cases to Stage 2.

## Data and preprocessing

### Dataset
We use the **NIH ChestX-ray14** dataset (Wang et al., 2017), which provides frontal-view CXR images and weak labels for 14 findings via the metadata file `Data_Entry_2017.csv`. Each study may contain multiple findings, so labels are represented as a 14-dimensional multi-hot vector.

### Train/validation/test splitting
When available, we follow the dataset’s official file lists (`train_val_list.txt`, `test_list.txt`). If official split files are not present, we form a random 80/20 train+val vs. test split.

To reduce patient-level leakage, we form the validation split at the **patient level** when a `Patient ID` column is available in the metadata. A fixed validation fraction (default 10%) is used.

### Image transforms
All images are converted to RGB and normalized with ImageNet statistics. We resize to $380\times 380$ (EfficientNet-B4 native input size) using a resize+crop pipeline:

- **Training transforms**: resize with margin, random resized crop (scale 0.9–1.0), random rotation (±10°), and mild brightness/contrast jitter.
- **Validation/test transforms**: resize with margin and deterministic center crop.

We avoid horizontal flips to preserve laterality cues.

## Stage 1: fast CNN screening with calibration

### Model architecture
Stage 1 is an EfficientNet-B4 backbone (Tan and Le, 2019; initialized from ImageNet pretraining) with a lightweight multi-label head:

- Feature extractor: EfficientNet-B4 without its original classification head.
- Head: dropout followed by a linear layer producing $C=14$ logits.

Let $z=f_\theta(x)\in\mathbb{R}^C$ be the raw logits.

### Training objective and imbalance handling
We optimize a class-weighted binary cross-entropy over logits:

$$
\mathcal{L}(z,y)=\sum_{c=1}^C \mathrm{BCEWithLogits}(z_c,y_c;\,\text{pos\_weight}_c),
$$

where $\text{pos\_weight}_c = \frac{N_{\text{neg},c}}{\max(N_{\text{pos},c},1)}$ is computed from the training split.

Optionally, a weighted sampler is used during training to mitigate per-image label sparsity.

### Two-phase fine-tuning schedule
Training is performed in two phases:

1. **Head warm-up**: freeze the backbone and train the head (and calibration parameter) with a higher learning rate.
2. **Full fine-tuning**: unfreeze the backbone and train all parameters with a lower learning rate and a validation-driven learning rate scheduler.

Gradient norms are clipped for stability.

### Post-hoc temperature scaling
To obtain calibrated probabilities, we apply scalar temperature scaling. A single parameter $T>0$ rescales logits:

$$
\tilde z = \frac{z}{T}, \qquad p = \sigma(\tilde z).
$$

We tune $T$ on the validation set by minimizing the negative log-likelihood (Guo et al., 2017; implemented as BCEWithLogitsLoss for the multi-label setting), updating only $T$ while keeping $\theta$ fixed.

## Adaptive routing to the VLM
Let $p_{\max}=\max_c p_c$ be the maximum calibrated probability across the 14 labels. The routing rule is:

$$
\text{route\_to\_stage2} = \mathbb{1}[p_{\max} < \tau],
$$

where $\tau\in(0,1)$ is a configurable threshold. High-confidence cases ($p_{\max}\ge\tau$) use Stage 1 only; low-confidence cases are forwarded to Stage 2.

This mechanism makes compute cost and latency tunable: larger $\tau$ invokes the VLM more often, while smaller $\tau$ increases Stage 1-only throughput.

## Stage 2: VLM second opinion with structured output

### VLM backend
For cases routed to Stage 2, we attempt to run **CheXagent** (Chen et al., 2024; a vision–language model for CXR interpretation). If the VLM cannot be loaded (e.g., missing dependencies or insufficient GPU memory), we fall back to a deterministic rule-based generator derived from Stage 1 outputs.

### Prompting and structured parsing
We use $K=2$ prompt templates to elicit consistent answers and to reduce prompt sensitivity. Each prompt requests output in a strict JSON schema:

```json
{
  "finding": "<primary finding or Normal>",
  "severity": "<normal|mild|moderate|severe>",
  "confidence": 0.0,
  "rationale": "<brief reasoning>",
  "secondary_findings": ["..."]
}
```

We run deterministic decoding (no sampling) and parse the generated JSON. Confidence values are clamped to $[0,1]$.

### Multi-prompt ensembling
Let the $k$-th prompt return a primary finding $\hat y^{(k)}_2$ and confidence $c^{(k)}_2$. We take the consensus finding by mode:

$$
\hat y_2 = \mathrm{mode}\big(\hat y^{(1)}_2,\ldots,\hat y^{(K)}_2\big),
$$

and compute the average confidence over prompts that match the consensus. We also report a **consensus strength** defined as the fraction of prompts agreeing with the final VLM finding. With $K=2$, ties can occur; in the current implementation ties are broken deterministically by prompt order and are reflected by lower consensus strength.

### Fallback rule-based Stage 2
When the VLM is unavailable, Stage 2 derives a structured response from Stage 1 probabilities. The top predicted label becomes the primary finding unless its probability indicates a normal case; severity is mapped by probability bins, and secondary findings are those above a fixed threshold.

## Reconciliation and conflict handling
Stage 1 produces a primary finding $\hat y_1 = \arg\max_c p_c$ with confidence $p^*=\max_c p_c$. Stage 2 returns $(\hat y_2, c_2)$.

### Fast path (Stage 2 skipped)
If $p_{\max}\ge\tau$, we return Stage 1’s primary finding and confidence directly and generate an explanation stating that the case was accepted by the calibrated CNN.

### Agreement: weighted fusion
If both stages run and the primary findings agree, we fuse confidences with fixed weights $w_1,w_2$:

$$
\hat y = \hat y_1, \qquad c = w_1\,p^* + w_2\,c_2,
$$

with $w_1=0.4$ and $w_2=0.6$ in our implementation. When available, we adopt Stage 2’s severity label; otherwise severity is mapped from $c$ using fixed bins.

### Disagreement: conflict flag and review
If $\hat y_1\neq\hat y_2$, we mark a **conflict** and attach a detailed explanation including both stage outputs and the VLM rationale. A radiologist review flag is raised for conflicting cases.

To quantify disagreement, we compute a confidence gap $\Delta = |p^* - c_2|$; this can be thresholded (e.g., $\Delta>\delta$ with $\delta=0.3$) to prioritize review workflows, while still retaining the full conflict trace for auditing.

### Secondary findings
Secondary findings are reported as the union of:

- Stage 1 findings with probability above a fixed threshold (0.3), excluding the primary, and
- Stage 2 `secondary_findings` when present.

## Feedback loop and continual calibration
To support continuous improvement, the system stores radiologist corrections in an append-only JSON-lines log. Each record contains the pipeline’s output, routing/conflict flags, optional Stage 1 logits, and optional ground-truth labels.

After accumulating a configured number of corrections (default 50) with both logits and labels, we update the temperature parameter $T$ using a lightweight search over candidate temperatures to minimize calibration loss on the feedback set. This provides continual calibration without retraining the backbone, making the system responsive to distribution shift or label drift in deployment settings.

## Implementation details
The system is implemented in Python using PyTorch for Stage 1, `timm` for EfficientNet backbones, and Hugging Face Transformers for VLM integration. Inference produces a fully structured JSON-serializable result containing per-class probabilities, routing decisions, reconciliation outcomes, and explanation text for auditability.
