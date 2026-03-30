"""
train.py — full training pipeline for EmotionNet.

Run from inside the backend/ folder:
    python train.py

What this script does:
  1. Loads the pre-processed face images from old_model/processed_data/
  2. Splits them 80 / 10 / 10 (train / val / test), reproducible with seed 42
  3. Computes per-channel mean & std from the TRAINING split only (correct practice)
  4. Applies strong data augmentation to the training split
  5. Handles class imbalance with a WeightedRandomSampler
  6. Trains EmotionNet with Adam + CosineAnnealingLR for up to 50 epochs
  7. Saves the best checkpoint (highest val accuracy) to emotion_model.pth
  8. Reports final test accuracy

Saved checkpoint format (emotion_model.pth):
  {
      "model_state_dict": ...,
      "class_names":      ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
      "mean":             [R, G, B],
      "std":              [R, G, B],
      "epoch":            <epoch at save>,
      "val_acc":          <val accuracy at save>,
  }

The mean/std are saved so the FastAPI server uses the exact same normalisation at inference,
ensuring no train-inference mismatch.
"""

import os
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Allow running this file from either the project root or the backend/ folder
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import EmotionNet

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(HERE, "..", "old_model", "processed_data")
SAVE_PATH = os.path.join(HERE, "emotion_model.pth")

BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Normalisation stats ──────────────────────────────────────────────────────
def compute_mean_std(subset: Subset, batch_size: int = 256):
    """
    Compute per-channel mean and std over a Subset of an ImageFolder dataset.

    All images are the same size (96×96), so averaging per-image channel means
    gives the exact same result as computing over every pixel. The std uses
    E[X²] − E[X]² for global accuracy.
    """
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    ch_mean = torch.zeros(3)
    ch_sq_mean = torch.zeros(3)
    n_images = 0

    for imgs, _ in loader:
        # imgs: (B, 3, H, W)
        flat = imgs.view(imgs.size(0), 3, -1)          # (B, 3, H*W)
        ch_mean    += flat.mean(dim=2).sum(dim=0)      # sum of per-image means
        ch_sq_mean += (flat ** 2).mean(dim=2).sum(dim=0)
        n_images   += imgs.size(0)

    ch_mean    /= n_images
    ch_sq_mean /= n_images
    ch_std = torch.sqrt((ch_sq_mean - ch_mean ** 2).clamp(min=1e-8))
    return ch_mean.tolist(), ch_std.tolist()


# ── Transforms ───────────────────────────────────────────────────────────────
def make_transform(mean, std, augment: bool = False) -> transforms.Compose:
    """
    Training augmentation:
      - Horizontal flip  (faces are symmetric)
      - ±10° rotation    (natural head tilts)
      - Colour jitter    (lighting variation between cameras / sessions)
      - Small translate  (slight off-centre crops)
    """
    ops = []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return transforms.Compose(ops)


# ── Class-balanced sampler ────────────────────────────────────────────────────
def build_weighted_sampler(subset: Subset) -> WeightedRandomSampler:
    """
    Each sample gets weight = 1 / class_count so that underrepresented emotions
    (e.g. disgust ~5 920 images) are sampled as often as happy (~11 398 images).
    """
    underlying = subset.dataset          # ImageFolder — has .targets
    targets = [underlying.targets[i] for i in subset.indices]
    counts = Counter(targets)
    weights = [1.0 / counts[t] for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(SEED)
    print(f"Device  : {DEVICE}")
    print(f"Data    : {os.path.abspath(DATA_DIR)}")
    print(f"Save to : {SAVE_PATH}\n")

    # ── 1. Load dataset with a minimal transform to compute stats ─────────────
    raw_ds = datasets.ImageFolder(DATA_DIR, transform=transforms.ToTensor())
    class_names: list = raw_ds.classes
    num_classes: int = len(class_names)
    total: int = len(raw_ds)

    print(f"Classes ({num_classes}) : {class_names}")
    print(f"Total images       : {total:,}")

    dist = Counter(raw_ds.targets)
    for cls, idx in raw_ds.class_to_idx.items():
        print(f"  {cls:<10} : {dist[idx]:,}")

    # ── 2. Deterministic split ─────────────────────────────────────────────────
    gen = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(total, generator=gen).tolist()

    train_size = int(0.80 * total)
    val_size   = int(0.10 * total)
    # test_size  = total - train_size - val_size   (remainder)

    train_idx = perm[:train_size]
    val_idx   = perm[train_size : train_size + val_size]
    test_idx  = perm[train_size + val_size:]

    print(f"\nSplit  — train: {len(train_idx):,}  val: {len(val_idx):,}  test: {len(test_idx):,}")

    # ── 3. Compute normalisation stats from training split only ──────────────
    print("\nComputing normalisation stats from training split…")
    train_raw = Subset(raw_ds, train_idx)
    mean, std = compute_mean_std(train_raw)
    print(f"  mean = {[round(m, 4) for m in mean]}")
    print(f"  std  = {[round(s, 4) for s in std]}")

    # ── 4. Build final split datasets with correct transforms ─────────────────
    def make_subset(augment: bool, indices: list) -> Subset:
        ds = datasets.ImageFolder(DATA_DIR, transform=make_transform(mean, std, augment=augment))
        return Subset(ds, indices)

    train_ds = make_subset(augment=True,  indices=train_idx)
    val_ds   = make_subset(augment=False, indices=val_idx)
    test_ds  = make_subset(augment=False, indices=test_idx)

    sampler      = build_weighted_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Batches — train: {len(train_loader)}  val: {len(val_loader)}  test: {len(test_loader)}\n")

    # ── 5. Model, loss, optimiser, scheduler ──────────────────────────────────
    model = EmotionNet(num_classes=num_classes).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"EmotionNet parameters : {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── 6. Training loop with early stopping ───────────────────────────────────
    best_val_acc = 0.0
    no_improve   = 0

    header = f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>8}  {'Best':>8}  {'LR':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── validate ──
        model.eval()
        correct = total_v = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total_v  += labels.size(0)

        val_acc  = correct / total_v * 100
        is_best  = val_acc > best_val_acc
        marker   = " ✓" if is_best else ""

        if is_best:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names":      class_names,
                    "mean":             mean,
                    "std":              std,
                    "epoch":            epoch,
                    "val_acc":          val_acc,
                },
                SAVE_PATH,
            )
        else:
            no_improve += 1

        print(
            f"{epoch:>5}  {avg_loss:>10.4f}  {val_acc:>7.2f}%  "
            f"{best_val_acc:>7.2f}%  {current_lr:>10.2e}{marker}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n[early stop] No improvement for {EARLY_STOP_PATIENCE} epochs — stopping.")
            break

    # ── 7. Final test evaluation using the best checkpoint ────────────────────
    print(f"\n{'=' * len(header)}")
    print(f"Best val accuracy : {best_val_acc:.2f}%  (saved → {SAVE_PATH})")

    checkpoint = torch.load(SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    correct = total_t = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_t += labels.size(0)

    test_acc = correct / total_t * 100
    print(f"Test accuracy     : {test_acc:.2f}%")
    print(f"{'=' * len(header)}\n")


if __name__ == "__main__":
    main()
