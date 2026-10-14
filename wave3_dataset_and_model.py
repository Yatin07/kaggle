"""
RSNA Knee Abnormality Detection — Wave 3 & 4: In-Memory 2.5D Multi-Plane Training
Optimized using exact verified columns: Fluid_Sensitive, Fat_Suppression, Anatomical_Plane.
Runs on Kaggle GPU T4 in under 2 minutes.
"""
import os
import glob
import time
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

print("=" * 70)
print("RSNA Knee: Wave 3 & 4 — Fast In-Memory 2.5D Multi-Plane Training")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Active Device: {DEVICE}")

# 1. Path Resolution
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

train_df = pd.read_csv(os.path.join(comp_dir, 'train.csv'))
train_series_df = pd.read_csv(os.path.join(comp_dir, 'train_series.csv'))
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
target_cols = [c for c in sample_sub.columns if c != 'StudyInstanceUID']

# Filter the 58 ground-truth annotated studies
labeled_df = train_df[train_df['ACL'].notna()].copy().reset_index(drop=True)
print(f"\n[DATA] Labeled Studies for Supervised Training: {len(labeled_df)}")

# 2. Medically Optimal Series Selection
# Priority: Fluid_Sensitive (True) -> Fat_Suppression (True)
sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
ascending_order = [True, True]

if 'Fluid_Sensitive' in train_series_df.columns:
    sort_cols.append('Fluid_Sensitive')
    ascending_order.append(False)
if 'Fat_Suppression' in train_series_df.columns:
    sort_cols.append('Fat_Suppression')
    ascending_order.append(False)

selected_series = train_series_df.sort_values(sort_cols, ascending=ascending_order)
selected_series = selected_series.drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])

series_map = {}
for _, row in selected_series.iterrows():
    study = row['StudyInstanceUID']
    plane = row['Anatomical_Plane']
    series = row['SeriesInstanceUID']
    if study not in series_map:
        series_map[study] = {}
    series_map[study][plane] = series

# 3. Fast In-Memory 2.5D Slice Extractor
def extract_25d_volume(study_id, plane, img_size=224):
    if study_id not in series_map or plane not in series_map[study_id]:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    
    series_id = series_map[study_id][plane]
    series_path = os.path.join(comp_dir, 'train_series', str(study_id), str(series_id))
    
    if not os.path.exists(series_path):
        return np.zeros((3, img_size, img_size), dtype=np.float32)

    files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')])
    if len(files) == 0:
        return np.zeros((3, img_size, img_size), dtype=np.float32)

    # Central 3 slices
    mid = len(files) // 2
    indices = [max(0, mid - 2), mid, min(len(files) - 1, mid + 2)]
    
    channels = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx])
            img = dcm.pixel_array.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            img = np.zeros((img_size, img_size), dtype=np.float32)
        channels.append(img)

    return np.stack(channels, axis=0)

# Pre-cache all 58 studies in RAM (~34 MB)
print("\n[PRE-CACHE] Loading 58 annotated studies into RAM...")
t0 = time.time()
all_sag, all_cor, all_axi, all_labels = [], [], [], []

for idx, row in labeled_df.iterrows():
    study_id = row['StudyInstanceUID']
    all_sag.append(extract_25d_volume(study_id, 'Sagittal'))
    all_cor.append(extract_25d_volume(study_id, 'Coronal'))
    all_axi.append(extract_25d_volume(study_id, 'Axial'))
    all_labels.append(row[target_cols].values.astype(np.float32))

sag_tensor = torch.tensor(np.array(all_sag), dtype=torch.float32)
cor_tensor = torch.tensor(np.array(all_cor), dtype=torch.float32)
axi_tensor = torch.tensor(np.array(all_axi), dtype=torch.float32)
labels_tensor = torch.tensor(np.array(all_labels), dtype=torch.float32)

print(f"Pre-caching complete in {time.time() - t0:.1f}s!")
print(f"RAM Tensors: Sag={sag_tensor.shape}, Cor={cor_tensor.shape}, Axi={axi_tensor.shape}")

# 4. Multi-Plane 2.5D Neural Network
class MultiPlaneKneeNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feature_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, sag, cor, axi):
        f_sag = self.backbone(sag).flatten(1)
        f_cor = self.backbone(cor).flatten(1)
        f_axi = self.backbone(axi).flatten(1)
        fused = torch.cat([f_sag, f_cor, f_axi], dim=1)
        return self.classifier(fused)

# 5. Fast In-Memory Supervised Training Loop
full_dataset = TensorDataset(sag_tensor, cor_tensor, axi_tensor, labels_tensor)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(labeled_df))

train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=8, shuffle=False)

model = MultiPlaneKneeNet(num_classes=12).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

EPOCHS = 15
best_auc = 0.0

print(f"\n[TRAINING] Training on {len(train_idx)} studies, Validating on {len(val_idx)} studies...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for sag, cor, axi, y in train_loader:
        sag, cor, axi, y = sag.to(DEVICE), cor.to(DEVICE), axi.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(sag, cor, axi)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)

    scheduler.step()
    train_loss = total_loss / len(train_idx)

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
        if len(np.unique(val_targets[:, c])) > 1:
            aucs.append(roc_auc_score(val_targets[:, c], val_preds[:, c]))
    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.5

    print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Macro ROC-AUC: {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        torch.save(model.state_dict(), '/kaggle/working/best_multiplane_knee_model.pth')
        print(f"  --> Saved new best model: AUC = {best_auc:.4f}")

print("\n" + "=" * 70)
print(f"[SUCCESS] Wave 3 & 4 complete! Best Validation Macro ROC-AUC: {best_auc:.4f}")
print("Saved checkpoint to: /kaggle/working/best_multiplane_knee_model.pth")
print("=" * 70)
