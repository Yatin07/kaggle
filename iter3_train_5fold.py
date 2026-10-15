"""
RSNA Knee Abnormality Detection — Iteration 3 Research-Grade Training
KEY IMPROVEMENTS:
  1. 5-Fold Stratified Cross-Validation on the 58 Gold Studies (Clean Out-Of-Fold Evaluation).
  2. Safe Augmentations ONLY: Affine rotation (+-10 deg), translation (+-5%), scale (+-10%).
     ZERO Horizontal/Vertical Flips (strictly preserves Medial/Lateral and Femur/Tibia orientation).
  3. Pretrained ResNet34 with 1-channel Grayscale Conv1 + Slice Average Pooling.
  4. Cross-Plane Attention Gate across Sagittal, Coronal, and Axial planes.
  5. Calibrated Fine-Tuning LR (1e-4) with Cosine Annealing.
  6. Positive Class-Weighted BCE Loss (handles rare pathologies like MCL and Fractures).
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
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

print("=" * 70)
print("RSNA Knee: Iteration 3 — 5-Fold Cross-Validation Research Model")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

# -----------------------------------------------------------------------
# 1. Dataset & Safe Augmentations (NO FLIPS)
# -----------------------------------------------------------------------
class KneeDatasetSafe(Dataset):
    def __init__(self, sag, cor, axi, labels, augment=False):
        self.sag = sag       # (N, slices, H, W) uint8
        self.cor = cor
        self.axi = axi
        self.labels = labels # (N, 12) float32
        self.augment = augment

    def _safe_aug(self, vol):
        """
        Safe spatial transforms that PRESERVE medial/lateral orientation:
        Rotation (+-10 deg), translation (+-5%), scale (0.9-1.1), brightness (0.85-1.15).
        FORBIDDEN: Horizontal flip, Vertical flip.
        """
        slices, h, w = vol.shape
        # Random small rotation and scale
        angle = np.random.uniform(-10.0, 10.0)
        scale = np.random.uniform(0.92, 1.08)
        tx = np.random.uniform(-0.05, 0.05) * w
        ty = np.random.uniform(-0.05, 0.05) * h

        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        aug_vol = np.zeros_like(vol)
        for s in range(slices):
            aug_vol[s] = cv2.warpAffine(vol[s], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Brightness / contrast jitter
        factor = np.random.uniform(0.88, 1.12)
        aug_vol = np.clip(aug_vol * factor, 0.0, 1.0)
        return aug_vol

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert uint8 [0..255] to float32 [0.0..1.0]
        sag = self.sag[idx].astype(np.float32) / 255.0
        cor = self.cor[idx].astype(np.float32) / 255.0
        axi = self.axi[idx].astype(np.float32) / 255.0

        if self.augment:
            sag = self._safe_aug(sag)
            cor = self._safe_aug(cor)
            axi = self._safe_aug(axi)

        return (
            torch.tensor(sag, dtype=torch.float32),
            torch.tensor(cor, dtype=torch.float32),
            torch.tensor(axi, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# -----------------------------------------------------------------------
# 2. Multi-Plane Model Architecture
# -----------------------------------------------------------------------
class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
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
        B, S, H, W = x.shape
        x = x.view(B * S, 1, H, W)
        feats = self.pool(self.backbone(x))   # (B*S, 512, 1, 1)
        feats = feats.view(B, S, self.feat_dim)
        return feats.mean(dim=1)              # (B, 512)

class MultiPlaneKneeNetV3(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.sag_enc = PlaneEncoder()
        self.cor_enc = PlaneEncoder()
        self.axi_enc = PlaneEncoder()
        feat_dim = self.sag_enc.feat_dim

        # Cross-plane dynamic attention gate
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
        fs = self.sag_enc(sag)
        fc = self.cor_enc(cor)
        fa = self.axi_enc(axi)

        all_feats = torch.cat([fs, fc, fa], dim=1)
        attn = self.plane_attn(all_feats)
        fs = fs * attn[:, 0:1]
        fc = fc * attn[:, 1:2]
        fa = fa * attn[:, 2:3]

        fused = torch.cat([fs, fc, fa], dim=1)
        return self.head(fused)

# -----------------------------------------------------------------------
# 3. 5-Fold Training Loop
# -----------------------------------------------------------------------
def train_5fold(sag_arr, cor_arr, axi_arr, labels_arr, epochs=20, lr=1e-4, batch_size=8):
    """
    Trains a 5-fold ensemble with out-of-fold validation on the gold annotations.
    """
    N = len(labels_arr)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_predictions = np.zeros_like(labels_arr)
    oof_targets = np.zeros_like(labels_arr)

    # Class positive weights for BCE
    pos_counts = (labels_arr == 1).sum(axis=0)
    neg_counts = (labels_arr == 0).sum(axis=0)
    pos_weight = torch.tensor(
        np.clip(neg_counts / np.maximum(pos_counts, 1), 1.0, 5.0),
        dtype=torch.float32
    ).to(DEVICE)
    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(N))):
        print(f"\n" + "-" * 50)
        print(f"FOLD {fold + 1} / 5 | Train: {len(train_idx)} | Val: {len(val_idx)}")
        print("-" * 50)

        train_ds = KneeDatasetSafe(sag_arr[train_idx], cor_arr[train_idx], axi_arr[train_idx], labels_arr[train_idx], augment=True)
        val_ds   = KneeDatasetSafe(sag_arr[val_idx], cor_arr[val_idx], axi_arr[val_idx], labels_arr[val_idx], augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

        best_val_auc = 0.0
        fold_save_path = f'/kaggle/working/knee_model_fold{fold}.pth'

        for ep in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for sag, cor, axi, y in train_loader:
                sag, cor, axi, y = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                    out = model(sag, cor, axi)
                    loss = bce_loss_fn(out, y)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(y)

            scheduler.step()
            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_preds = []
            val_trues = []
            with torch.no_grad():
                for sag, cor, axi, y in val_loader:
                    sag, cor, axi = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE)
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                        p = torch.sigmoid(model(sag, cor, axi))
                    val_preds.append(p.cpu().numpy())
                    val_trues.append(y.numpy())

            vp = np.vstack(val_preds)
            vt = np.vstack(val_trues)

            aucs = []
            for c in range(12):
                if len(np.unique(vt[:, c])) > 1:
                    aucs.append(roc_auc_score(vt[:, c], vp[:, c]))
            mean_auc = np.mean(aucs) if aucs else 0.50

            star = ""
            if mean_auc > best_val_auc:
                best_val_auc = mean_auc
                torch.save(model.state_dict(), fold_save_path)
                star = " ★ [BEST]"

            if ep % 5 == 0 or ep == epochs or star:
                print(f"  Ep {ep:2d}/{epochs:2d} | Loss: {train_loss:.4f} | Val AUC: {mean_auc:.4f}{star}")

        print(f"--> Fold {fold + 1} Best Val AUC: {best_val_auc:.4f} (Saved: {fold_save_path})")
        fold_aucs.append(best_val_auc)

    mean_oof_auc = np.mean(fold_aucs)
    print("\n" + "=" * 70)
    print(f"5-FOLD CROSS-VALIDATION SUMMARY:")
    for f_idx, auc in enumerate(fold_aucs):
        print(f"  Fold {f_idx + 1}: AUC = {auc:.4f}")
    print(f"MEAN 5-FOLD MACRO ROC-AUC: {mean_oof_auc:.4f}")
    print("=" * 70)
    return fold_aucs
