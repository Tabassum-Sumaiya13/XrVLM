"""
Stage 1 — EfficientNet-B4 fast screening classifier.

Outputs raw logits (14 classes). No sigmoid in forward pass —
BCEWithLogitsLoss handles it during training.

Includes temperature scaling as a lightweight calibration layer.
"""

import torch
import torch.nn as nn
import timm

from config import NUM_CLASSES, STAGE1_BACKBONE, STAGE1_DROPOUT, TEMPERATURE_INIT


# ══════════════════════════════════════════════════════════════════════════════
# Temperature Scaling
# ══════════════════════════════════════════════════════════════════════════════
class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration: divides logits by a learned temperature T
    before softmax / sigmoid. T > 1 softens predictions (reduces
    overconfidence), T < 1 sharpens them.

    Only one parameter — can be tuned on a held-out calibration set
    without touching the backbone weights.
    """

    def __init__(self, init_temp: float = TEMPERATURE_INIT):
        super().__init__()
        # Store as log so that exp() keeps T strictly positive
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temp, dtype=torch.float32))
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits: logits / T"""
        return logits / self.temperature


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 Model (EfficientNet-B4 + Temperature Scaling)
# ══════════════════════════════════════════════════════════════════════════════
class Stage1Model(nn.Module):
    """
    EfficientNet-B4 backbone with:
      • Custom 14-class multi-label head
      • Integrated temperature scaling for calibrated confidence
    """

    def __init__(
        self,
        backbone_name: str = STAGE1_BACKBONE,
        num_classes: int = NUM_CLASSES,
        dropout: float = STAGE1_DROPOUT,
        pretrained: bool = True,
        init_temp: float = TEMPERATURE_INIT,
    ):
        super().__init__()

        # Load EfficientNet-B4 backbone from timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove original head → feature extractor
            drop_rate=dropout,
        )
        self.feature_dim = self.backbone.num_features  # 1792 for B4

        # Custom multi-label classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Temperature scaling (post-hoc calibration)
        self.temp_scaler = TemperatureScaler(init_temp=init_temp)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from backbone."""
        return self.backbone(x)  # (B, 1792)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logits before temperature scaling."""
        features = self.forward_features(x)
        return self.classifier(features)  # (B, 14)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward: image → features → logits → temp-scaled logits.
        Returns SCALED logits (use sigmoid for probabilities).
        """
        logits = self.forward_logits(x)
        return self.temp_scaler(logits)

    def get_calibrated_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: returns calibrated sigmoid probabilities."""
        with torch.no_grad():
            scaled_logits = self.forward(x)
            return torch.sigmoid(scaled_logits)


# ── fine-tuning helpers ──────────────────────────────────────────────────────
def freeze_backbone(model: Stage1Model) -> None:
    """Freeze backbone; only classifier + temp_scaler stay trainable."""
    for name, param in model.named_parameters():
        if "classifier" not in name and "temp_scaler" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[freeze_backbone] trainable: {trainable:,} / {total:,}")


def unfreeze_backbone(model: Stage1Model) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters())
    print(f"[unfreeze_backbone] trainable: {trainable:,}")


def build_stage1_model(
    pretrained: bool = True,
    dropout: float = STAGE1_DROPOUT,
    init_temp: float = TEMPERATURE_INIT,
) -> Stage1Model:
    """Factory function for Stage 1."""
    return Stage1Model(
        backbone_name=STAGE1_BACKBONE,
        num_classes=NUM_CLASSES,
        dropout=dropout,
        pretrained=pretrained,
        init_temp=init_temp,
    )


# ── calibration tuning ──────────────────────────────────────────────────────
def tune_temperature(
    model: Stage1Model,
    val_loader,
    device: torch.device,
    lr: float = 0.01,
    max_iter: int = 100,
) -> float:
    """
    Optimise the temperature parameter on a validation set.

    Uses NLL loss (cross-entropy after temperature scaling).
    Only the temperature parameter is updated — backbone is frozen.
    Returns the final temperature value.
    """
    # Freeze everything except temperature
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.temp_scaler.log_temperature.requires_grad = True

    nll_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(
        [model.temp_scaler.log_temperature], lr=lr, max_iter=max_iter
    )

    # Collect all logits and labels first (memory permitting)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model.forward_logits(images)
            all_logits.append(logits)
            all_labels.append(labels.to(device))

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    def _eval():
        optimizer.zero_grad()
        scaled = model.temp_scaler(all_logits)
        loss = nll_criterion(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(_eval)

    final_T = model.temp_scaler.temperature.item()
    print(f"[tune_temperature] Final T = {final_T:.4f}")

    # Re-enable all parameters
    for param in model.parameters():
        param.requires_grad = True

    return final_T


# ── smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import get_device

    device = get_device()
    print(f"Device: {device}\n")

    model = build_stage1_model(pretrained=True)
    print(f"Feature dim: {model.feature_dim}")
    print(f"Temperature: {model.temp_scaler.temperature.item():.4f}")

    freeze_backbone(model)
    unfreeze_backbone(model)

    model = model.to(device)
    dummy = torch.randn(2, 3, 380, 380, device=device)
    with torch.no_grad():
        logits = model(dummy)
        probs = torch.sigmoid(logits)
    print(f"\nForward: input {dummy.shape} → logits {logits.shape}")
    print(f"Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"Temperature: {model.temp_scaler.temperature.item():.4f}")
