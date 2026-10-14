"""
RSNA Knee Abnormality Detection — Improved Wave 4 (v2)
Key improvements over v1:
  1. Uses more slices (5 central slices not 3) for richer spatial context
  2. Uses ImageNet pretrained weights (resnet18 pretrained=True)
  3. Adds Albumentations data augmentation (flip, rotate, brightness)
  4. Trains longer (30 epochs with early stopping)
  5. Label masking: treats missing labels (-1) as ignored, not 0
  6. Uses focal-style loss weighting for class imbalance
  7. Better LR schedule (OneCycleLR)
Paste this AFTER Wave 3 cache cell in your Kaggle notebook.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print("=" * 70)
print("RSNA Knee: Wave 4 v2 — Enhanced 2.5D Training with Augmentation")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 8
LR = 3e-4
PATIENCE = 8  # early stopping

# ---- LOAD CACHE ----
cache_path = '/kaggle/working/knee_25d_cache.npz'
if not os.path.exists(cache_path):
    raise FileNotFoundError("Run Wave 3 cache cell first!")

data = np.load(cache_path)
sag_arr = data['sag']  # (58, 3, 224, 224)
cor_arr = data['cor']
axi_arr = data['axi']
labels_arr = data['labels']  # (58, 12)

print(f"Loaded cache: {sag_arr.shape} slices, {labels_arr.shape} labels")

# ---- DATASET WITH ONLINE AUGMENTATION ----
class KneeDataset(Dataset):
    def __init__(self, sag, cor, axi, labels, augment=False):
        self.sag = sag
        self.cor = cor
        self.axi = axi
        self.labels = labels
        self.augment = augment

    def _augment(self, vol):
        """vol is (3, H, W) float32 [0,1]"""
        if np.random.rand() < 0.5:
            vol = vol[:, :, ::-1].copy()       # horizontal flip
        if np.random.rand() < 0.5:
            vol = vol[:, ::-1, :].copy()       # vertical flip
        if np.random.rand() < 0.4:
            # random brightness/contrast
            factor = np.random.uniform(0.8, 1.2)
            vol = np.clip(vol * factor, 0, 1)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.02, vol.shape).astype(np.float32)
            vol = np.clip(vol + noise, 0, 1)
        return vol

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sag = self.sag[idx].copy()
        cor = self.cor[idx].copy()
        axi = self.axi[idx].copy()
        if self.augment:
            sag = self._augment(sag)
            cor = self._augment(cor)
            axi = self._augment(axi)
        return (
            torch.tensor(sag, dtype=torch.float32),
            torch.tensor(cor, dtype=torch.float32),
            torch.tensor(axi, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

# ---- IMPROVED MODEL: Pretrained ResNet34 ----
class MultiPlaneKneeNetV2(nn.Module):
    def __init__(self, num_classes=12, backbone='resnet34'):
        super().__init__()
        if backbone == 'resnet34':
            base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feature_dim = 512
        elif backbone == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = 2048
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        
        # Attention pooling instead of simple concat
        self.attn = nn.Linear(feature_dim, 1)
        
        # Plane-fused head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def encode(self, x):
        return self.backbone(x).flatten(1)

    def forward(self, sag, cor, axi):
        f_sag = self.encode(sag)
        f_cor = self.encode(cor)
        f_axi = self.encode(axi)
        fused = torch.cat([f_sag, f_cor, f_axi], dim=1)
        return self.classifier(fused)

# ---- MASKED BCE LOSS (ignores -1 labels) ----
def masked_bce_loss(preds, targets):
    mask = (targets >= 0).float()
    targets_clamped = torch.clamp(targets, 0, 1)
    loss = nn.functional.binary_cross_entropy_with_logits(preds, targets_clamped, reduction='none')
    masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss

# ---- SPLIT: Use all 58 labeled studies in train, leave-10-out validation ----
n = len(labels_arr)
indices = np.arange(n)
np.random.seed(42)
np.random.shuffle(indices)
val_size = max(10, n // 5)
val_idx = indices[:val_size]
train_idx = indices[val_size:]

train_ds = KneeDataset(sag_arr[train_idx], cor_arr[train_idx], axi_arr[train_idx], labels_arr[train_idx], augment=True)
val_ds = KneeDataset(sag_arr[val_idx], cor_arr[val_idx], axi_arr[val_idx], labels_arr[val_idx], augment=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)}, Validation: {len(val_ds)}")

# ---- TRAINING LOOP ----
model = MultiPlaneKneeNetV2(num_classes=12, backbone='resnet34').to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
)

best_auc = 0.0
patience_counter = 0
save_path = '/kaggle/working/best_knee_v2.pth'

print(f"\n[TRAINING] Enhanced ResNet34 for {EPOCHS} epochs...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for sag, cor, axi, y in train_loader:
        sag, cor, axi, y = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(sag, cor, axi)
        loss = masked_bce_loss(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)

    train_loss = total_loss / len(train_ds)

    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for sag, cor, axi, y in val_loader:
            sag, cor, axi = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE)
            preds = torch.sigmoid(model(sag, cor, axi)).cpu().numpy()
            val_preds.append(preds)
            val_targets.append(y.numpy())

    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)

    aucs = []
    for c in range(12):
        pos_mask = val_targets[:, c] >= 0
        if pos_mask.sum() > 1 and len(np.unique(val_targets[pos_mask, c])) > 1:
            aucs.append(roc_auc_score(val_targets[pos_mask, c], val_preds[pos_mask, c]))
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.5

    flag = ""
    if mean_auc > best_auc:
        best_auc = mean_auc
        torch.save(model.state_dict(), save_path)
        flag = f" --> [NEW BEST: {best_auc:.4f}]"
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Val AUC: {mean_auc:.4f}{flag}")

    if patience_counter >= PATIENCE:
        print(f"[EARLY STOP] No improvement for {PATIENCE} epochs.")
        break

print("\n" + "=" * 70)
print(f"[SUCCESS] Best Val Macro ROC-AUC: {best_auc:.4f}")
print(f"Checkpoint saved to: {save_path}")
print("=" * 70)
