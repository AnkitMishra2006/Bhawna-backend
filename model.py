"""
EmotionNet — improved CNN for 7-class facial emotion recognition.

Architecture improvements over the old 3-conv model:
  • 4 conv blocks instead of 3  (deeper representation)
  • 2 convolutions per block     (more capacity per block)
  • BatchNorm after every conv   (faster training, reduces internal covariate shift)
  • Dropout2d in conv blocks     (regularisation — drops entire feature maps)
  • Dropout before FC            (prevents co-adaptation of neurons)
  • Global Average Pooling       (removes spatial inductive bias, fewer params than flatten)
  • bias=False on conv layers    (BN makes bias redundant — saves memory)
  • predict_proba() method       (returns softmax probabilities 0–1 per class)

Input:  3 × 96 × 96  (normalised RGB face crop)
Output: 7 logits     (angry, disgust, fear, happy, neutral, sad, surprise)

Parameter count: ~1.24 M  (vs 88 K in the old model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, dropout: float = 0.25) -> nn.Sequential:
    """Two conv-BN-ReLU layers followed by MaxPool and spatial Dropout."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),        # spatial dims halved
        nn.Dropout2d(dropout),  # drops full feature-map channels
    )


class EmotionNet(nn.Module):
    """
    96×96 input → 4 progressively deeper conv blocks → Global Average Pool
    → two-layer MLP classifier → 7 logits.

    Spatial resolution at each stage (96×96 input):
        after block 1 : 48×48×32
        after block 2 : 24×24×64
        after block 3 : 12×12×128
        after block 4 :  6×6×256
        after GAP     :    256
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            _conv_block(3,   32),   # 96 → 48
            _conv_block(32,  64),   # 48 → 24
            _conv_block(64,  128),  # 24 → 12
            _conv_block(128, 256),  # 12 →  6
        )

        # Collapse spatial dims to a single vector; works for any input size.
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_classes)."""
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities, shape (batch, num_classes).
        Use this during inference — never during training (no gradients)."""
        return F.softmax(self.forward(x), dim=1)
