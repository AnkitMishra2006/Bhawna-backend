"""
Bhawna — improved CNN for 7-class facial emotion recognition.

Architecture:
  • 4 conv blocks instead of 3           (deeper representation)
  • 2 convolutions per block             (more capacity per block)
  • BatchNorm after every conv           (faster training, reduces internal covariate shift)
  • Squeeze-and-Excitation after each
    conv pair                            (learnt channel attention — see SEBlock)
  • Block 1 uses a strided Conv2d        (learnt downsampling preserves fine detail)
    instead of MaxPool2d                 that fixed MaxPool always discards
  • Blocks 3 & 4 use depthwise-         (same receptive field at ~1/8th the MACs)
    separable second convolution         also acts as an implicit regulariser
  • Dropout2d in conv blocks             (regularisation — drops entire feature maps)
  • 3-layer MLP classifier              (256 → 512 → 128 → 7, more separation capacity)
  • Dropout before FC layers             (prevents co-adaptation of neurons)
  • Global Average Pooling               (removes spatial inductive bias, fewer params)
  • bias=False on conv layers            (BN makes bias redundant — saves memory)
  • predict_proba() method               (returns softmax probabilities 0–1 per class)

Input:  3 × 96 × 96  (normalised RGB face crop)
Output: 7 logits     (angry, disgust, fear, happy, neutral, sad, surprise)

Parameter count: ~1.35 M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018 — arxiv.org/abs/1709.01507).

    After each pair of conv-BN-ReLU layers the SE block learns which feature-map
    channels are most informative for the current input and rescales them:

      1. Squeeze  — global average pool collapses (B, C, H, W) → (B, C) scalar per channel
      2. Excite   — a tiny 2-layer MLP (C → C//16 → C) outputs a weight in [0, 1]
      3. Rescale  — each feature map is element-wise multiplied by its learned weight

    For emotion recognition this is especially valuable: different emotions rely on
    different facial regions.  SE lets the network amplify channels that encode
    eye shape (fear/surprise) or mouth curvature (happy/sad) and suppress channels
    encoding irrelevant background texture — adaptively, per input frame.

    reduction=16 is the standard ratio from the original SENet paper.  The floor
    at 4 keeps the bottleneck meaningful even for the first (32-channel) block.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)       # floor prevents degenerate squeeze
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),                             # (B, C)
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),                             # per-channel weight in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)  # broadcast over H × W
        return x * scale


def _dws_conv(channels: int) -> nn.Sequential:
    """
    Depthwise-separable convolution (Howard et al., 2017 — MobileNets).

    Replaces a standard 3×3 Conv2d(C, C) with:
      1. Depthwise  — Conv2d(C, C, 3, groups=C)  : each channel filtered independently
      2. Pointwise  — Conv2d(C, C, 1)             : mixes information across channels

    This has the same receptive field and output shape as a standard 3×3 conv but
    uses only C*(9 + C) parameters vs C*C*9 for the dense conv — a ~8× reduction
    at typical channel counts (128, 256).  The reduced parameter count acts as an
    additional regulariser for deeper blocks where over-parameterisation is more
    likely.  BN+ReLU after each step follows MobileNetV2 best practice.
    """
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                  groups=channels, bias=False),   # depthwise
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),  # pointwise
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


def _conv_block(
    in_ch: int,
    out_ch: int,
    dropout: float = 0.25,
    use_strided_downsample: bool = False,
    use_dws: bool = False,
) -> nn.Sequential:
    """
    Two conv-BN-ReLU layers → SE channel attention → downsample → spatial Dropout.

    use_strided_downsample=True  (block 1):
        Replaces MaxPool2d(2) with a strided Conv2d(out_ch, out_ch, 3, stride=2).
        A learned downsampling retains more information than always picking the
        maximum — especially important in the first block where features are still
        low-level edges and textures at full 96×96 resolution.

    use_dws=True  (blocks 3 & 4):
        The SECOND conv-BN-ReLU pair is replaced by a depthwise-separable
        conv (see _dws_conv).  The first conv is kept dense to ensure complete
        channel mixing when moving from in_ch→out_ch.
    """
    # First conv: always dense (in_ch → out_ch)
    layers: list = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]

    # Second conv: dense or depthwise-separable
    if use_dws:
        layers += list(_dws_conv(out_ch).children())   # unpack the Sequential
    else:
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]

    layers.append(SEBlock(out_ch))        # channel attention

    # Spatial downsampling
    if use_strided_downsample:
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    else:
        layers.append(nn.MaxPool2d(2))

    layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class EmotionNet(nn.Module):
    """
    96×96 input → 4 progressively deeper conv blocks → Global Average Pool
    → three-layer MLP classifier → 7 logits.

    Spatial resolution at each stage (96×96 input):
        after block 1 : 48×48×32   (strided conv downsampling — learnt)
        after block 2 : 24×24×64   (MaxPool)
        after block 3 : 12×12×128  (MaxPool, DWS second conv)
        after block 4 :  6×6×256   (MaxPool, DWS second conv)
        after GAP     :    256

    The three-layer classifier (256 → 512 → 128 → 7) gives the network more
    capacity to separate the 7 emotion classes after the convolutional feature
    extraction.  The wider 512-unit hidden layer can represent more complex
    combinations of facial features, while the bottleneck to 128 before the
    output layer acts as a final regulariser.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: strided conv instead of MaxPool to learn the downsampling
            _conv_block(3,   32,  use_strided_downsample=True),   # 96 → 48
            _conv_block(32,  64),                                  # 48 → 24
            # Blocks 3 & 4: depthwise-separable second conv for efficiency + regularisation
            _conv_block(64,  128, use_dws=True),                   # 24 → 12
            _conv_block(128, 256, use_dws=True),                   # 12 →  6
        )

        # Collapse spatial dims to a single vector; works for any input size.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Three-layer MLP: wider hidden layer for better class separation,
        # then bottleneck to 128 as a final regulariser before 7-class output.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # ── Proper weight initialisation (He et al., 2015) ──────────────────
        # Every major CNN reference implementation (ResNet, EfficientNet,
        # MobileNet) explicitly applies Kaiming Normal init for conv layers
        # followed by ReLU.  PyTorch's default (Kaiming Uniform) is okay, but
        # Normal with mode='fan_out' is the established best practice:
        #   • fan_out preserves variance in the backward pass (better gradient
        #     flow through deep networks)
        #   • Normal distribution matches the theoretical derivation in the
        #     He et al. paper more closely than Uniform
        # BN layers: weight=1, bias=0 (already the default, but explicit is
        # defensive against future PyTorch changes).
        # Linear layers: Kaiming Normal is also correct for ReLU activations.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
