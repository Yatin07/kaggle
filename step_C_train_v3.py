"""
RSNA Knee — Improvement Module C: GPU-Optimized Full Training (v3)
OPTIMIZATIONS FOR DUAL T4 GPUs:
  1. Multi-GPU DataParallel: detects and utilizes ALL available GPUs (T4 x2)
  2. Mixed Precision Training (torch.cuda.amp): 2-3x speedup via Tensor Cores + 50% VRAM saving
  3. Optimized DataLoader: num_workers=4, pin_memory=True, non_blocking=True
  4. Dynamic Batch Sizing: scaled according to available GPUs (Batch Size 8 or 16)
  5. Robust Pretrained Fallback: gracefully handles offline Kaggle kernel without crash
  6. Focal Masked BCE Loss + Plane Attention Gate + OneCycleLR + Early Stopping
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

print("=" * 70)
print("RSNA Knee: Step C — GPU-Optimized Multi-GPU Training (AMP + DataParallel)")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. GPU & Compute Device Setup
# -----------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

print(f"Primary Compute Device: {DEVICE}")
print(f"Total GPUs Available:   {NUM_GPUS}")
for i in range(NUM_GPUS):
    gpu_name = torch.cuda.get_device_name(i)
    vram_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  --> GPU {i}: {gpu_name} ({vram_gb:.1f} GB VRAM)")

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Auto-tunes fastest convolution algorithms

# Batch size scales with GPU count
BASE_BATCH = 8
BATCH_SIZE = BASE_BATCH * max(1, NUM_GPUS)
EPOCHS = 25
LR = 3e-4 * max(1, NUM_GPUS)
PATIENCE = 6
IMG_SIZE = 224

TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

# -----------------------------------------------------------------------
# 2. Load Full Studies Cache
# -----------------------------------------------------------------------
cache_path = '/kaggle/working/all_studies_cache.npz'
fallback_path = '/kaggle/working/knee_25d_cache.npz'

if os.path.exists(cache_path):
    data = np.load(cache_path)
    print(f"\n[CACHE] Loaded full multi-study cache: {cache_path}")
elif os.path.exists(fallback_path):
    print(f"\n[CACHE WARNING] Full cache not found. Using baseline cache: {fallback_path}")
    data = np.load(fallback_path)
else:
    raise FileNotFoundError("No cache file found! Please execute step_B_cache_v2.py first.")

sag_arr    = data['sag']     # (N, slices, 224, 224)
cor_arr    = data['cor']
axi_arr    = data['axi']
labels_arr = data['labels']  # (N, 12)

N = len(labels_arr)
N_SLICES = sag_arr.shape[1]
print(f"Total Studies in Cache: {N}")
print(f"Slices per plane:       {N_SLICES}")
print(f"Effective Batch Size:   {BATCH_SIZE} across {max(1, NUM_GPUS)} GPU(s)")

# Count gold labeled studies
gold_count = np.sum(np.all(labels_arr >= 0, axis=1))
print(f"Gold Ground-Truth Studies: {gold_count}/{N}")

# -----------------------------------------------------------------------
# 3. Dataset & Fast PyTorch Augmentations
# -----------------------------------------------------------------------
class KneeDatasetV3(Dataset):
    def __init__(self, sag, cor, axi, labels, augment=False):
        self.sag = sag
        self.cor = cor
        self.axi = axi
        self.labels = labels
        self.augment = augment

    def _aug(self, vol):
        # vol: (slices, H, W)
        if np.random.rand() < 0.5:
            vol = vol[:, :, ::-1].copy()  # Horizontal flip (anatomically preserved)
        if np.random.rand() < 0.3:
            vol = vol[:, ::-1, :].copy()  # Vertical flip
        if np.random.rand() < 0.4:
            factor = np.random.uniform(0.85, 1.15)
            vol = np.clip(vol * factor, 0.0, 1.0)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.015, vol.shape).astype(np.float32)
            vol = np.clip(vol + noise, 0.0, 1.0)
        return vol

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sag = self.sag[idx].astype(np.float32)
        cor = self.cor[idx].astype(np.float32)
        axi = self.axi[idx].astype(np.float32)

        # Scale uint8 [0..255] to [0.0..1.0]
        if self.sag.dtype == np.uint8 or sag.max() > 1.0:
            sag = sag / 255.0
            cor = cor / 255.0
            axi = axi / 255.0

        if self.augment:
            sag = self._aug(sag)
            cor = self._aug(cor)
            axi = self._aug(axi)
        return (
            torch.tensor(sag, dtype=torch.float32),
            torch.tensor(cor, dtype=torch.float32),
            torch.tensor(axi, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# -----------------------------------------------------------------------
# 4. Multi-Plane Deep Neural Network with Attention Gates
# -----------------------------------------------------------------------
from torchvision import models

class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Try loading pretrained EfficientNet-B3; fallback gracefully if offline without cache
        loaded = False
        try:
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
            old_conv = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            with torch.no_grad():
                base.features[0][0].weight.copy_(old_conv.weight[:, :1, :, :])
            self.backbone = base.features
            self.pool = base.avgpool
            self.feat_dim = 1536
            loaded = True
        except Exception:
            pass

        if not loaded:
            # High-efficiency ResNet34 fallback
            try:
                base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                base = models.resnet34(weights=None)
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                base.conv1.weight.copy_(old_conv.weight[:, :1, :, :])
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feat_dim = 512

    def forward(self, x):
        # x: (B, S, H, W)
        B, S, H, W = x.shape
        x = x.view(B * S, 1, H, W)
        feats = self.pool(self.backbone(x))  # (B*S, feat_dim, 1, 1)
        feats = feats.view(B, S, self.feat_dim)
        return feats.mean(dim=1)             # (B, feat_dim) pooled across slices

class MultiPlaneKneeNetV3(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.sag_enc = PlaneEncoder()
        self.cor_enc = PlaneEncoder()
        self.axi_enc = PlaneEncoder()
        feat_dim = self.sag_enc.feat_dim

        # Dynamic cross-plane attention gate
        self.plane_attn = nn.Sequential(
            nn.Linear(feat_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, sag, cor, axi):
        fs = self.sag_enc(sag)   # (B, feat_dim)
        fc = self.cor_enc(cor)
        fa = self.axi_enc(axi)

        # Cross-plane weighting
        all_feats = torch.cat([fs, fc, fa], dim=1)
        attn = self.plane_attn(all_feats) # (B, 3)
        fs = fs * attn[:, 0:1]
        fc = fc * attn[:, 1:2]
        fa = fa * attn[:, 2:3]

        fused = torch.cat([fs, fc, fa], dim=1)
        return self.head(fused)

# -----------------------------------------------------------------------
# 5. Masked Focal BCE Loss
# -----------------------------------------------------------------------
def focal_masked_bce(preds, targets, gamma=2.0, alpha=0.75):
    """Focal Loss with masking: ignores targets where label == -1"""
    mask = (targets >= 0).float()
    t = torch.clamp(targets, 0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(preds, t, reduction='none')
    p = torch.sigmoid(preds)
    pt = p * t + (1.0 - p) * (1.0 - t)
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = alpha * t + (1.0 - alpha) * (1.0 - t)
    focal_bce = alpha_weight * focal_weight * bce
    return (focal_bce * mask).sum() / (mask.sum() + 1e-8)

# -----------------------------------------------------------------------
# 6. Train/Val Split (Hold out gold cases for pristine validation)
# -----------------------------------------------------------------------
gold_mask = np.all(labels_arr >= 0, axis=1)
gold_indices = np.where(gold_mask)[0]
pseudo_indices = np.where(~gold_mask)[0]

np.random.seed(42)
np.random.shuffle(pseudo_indices)

val_pseudo_count = max(20, len(pseudo_indices) // 10)
val_idx = np.concatenate([gold_indices, pseudo_indices[:val_pseudo_count]])
train_idx = pseudo_indices[val_pseudo_count:]

print(f"\n[DATA SPLIT] Train: {len(train_idx)} studies | Validation: {len(val_idx)} studies (includes all {len(gold_indices)} gold)")

num_workers = min(4, os.cpu_count() or 2)
train_ds = KneeDatasetV3(sag_arr[train_idx], cor_arr[train_idx], axi_arr[train_idx], labels_arr[train_idx], augment=True)
val_ds   = KneeDatasetV3(sag_arr[val_idx], cor_arr[val_idx], axi_arr[val_idx], labels_arr[val_idx], augment=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda'),
    drop_last=True
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_workers, pin_memory=(DEVICE.type == 'cuda')
)

# -----------------------------------------------------------------------
# 7. Model Instantiation & Multi-GPU Setup
# -----------------------------------------------------------------------
raw_model = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)

# Wrap with DataParallel if dual T4 GPUs are available
if NUM_GPUS > 1:
    print(f"\n[MULTI-GPU] Activating torch.nn.DataParallel across {NUM_GPUS} GPUs!")
    model = nn.DataParallel(raw_model)
else:
    model = raw_model

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=EPOCHS,
    steps_per_epoch=max(1, len(train_loader)),
    pct_start=0.15
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

# -----------------------------------------------------------------------
# 8. High-Performance GPU Training Loop with AMP
# -----------------------------------------------------------------------
best_auc = 0.0
patience_counter = 0
save_path = '/kaggle/working/best_knee_v3.pth'

print(f"\n[TRAINING] Starting {EPOCHS} epochs with Mixed Precision (AMP) on {DEVICE}...")

for epoch in range(1, EPOCHS + 1):
    t_start = time.time()
    model.train()
    running_loss, sample_count = 0.0, 0

    for sag, cor, axi, y in train_loader:
        sag = sag.to(DEVICE, non_blocking=True)
        cor = cor.to(DEVICE, non_blocking=True)
        axi = axi.to(DEVICE, non_blocking=True)
        y   = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Autocast enables FP16 Tensor Cores on T4 GPUs
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            preds = model(sag, cor, axi)
            loss = focal_masked_bce(preds, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * len(y)
        sample_count += len(y)

    train_loss = running_loss / max(sample_count, 1)

    # Fast Validation with Test-Time Augmentation (TTA)
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for sag, cor, axi, y in val_loader:
            sag = sag.to(DEVICE, non_blocking=True)
            cor = cor.to(DEVICE, non_blocking=True)
            axi = axi.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                p1 = torch.sigmoid(model(sag, cor, axi))
                # TTA horizontal flip
                p2 = torch.sigmoid(model(sag.flip(-1), cor.flip(-1), axi.flip(-1)))
                p_avg = (p1 + p2) * 0.5

            val_preds.append(p_avg.cpu().numpy())
            val_targets.append(y.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    # Compute Macro ROC-AUC specifically on clean gold annotations
    clean_val_mask = np.all(val_targets >= 0, axis=1)
    if clean_val_mask.sum() > 5:
        vp, vt = val_preds[clean_val_mask], val_targets[clean_val_mask]
    else:
        vp, vt = val_preds, val_targets

    aucs = []
    for c in range(12):
        valid = vt[:, c] >= 0
        if valid.sum() > 2 and len(np.unique(vt[valid, c])) > 1:
            aucs.append(roc_auc_score(vt[valid, c], vp[valid, c]))
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.50

    elapsed = time.time() - t_start

    # GPU telemetry
    gpu_mem = f"VRAM: {torch.cuda.memory_allocated() / 1e6:.0f}MB" if DEVICE.type == 'cuda' else ""
    star = ""

    if mean_auc > best_auc:
        best_auc = mean_auc
        patience_counter = 0
        # Unwrap DataParallel before saving weights
        weights = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(weights, save_path)
        star = " ★ [BEST]"
    else:
        patience_counter += 1

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val AUC: {mean_auc:.4f} | {elapsed:.1f}s | {gpu_mem}{star}")

    if patience_counter >= PATIENCE:
        print(f"\n[EARLY STOP] Validation AUC did not improve for {PATIENCE} epochs.")
        break

print("\n" + "=" * 70)
print(f"[TRAINING COMPLETE] Best Validation Macro ROC-AUC: {best_auc:.4f}")
print(f"[CHECKPOINT SAVED] {save_path}")
print("=" * 70)
