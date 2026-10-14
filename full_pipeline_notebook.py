"""
RSNA Knee Abnormality Detection — Consolidated 2.5D Multi-Plane Pipeline
All Waves (Cache -> Train -> Predict -> Submit) in a single standalone script.
Settings: GPU T4, Internet: OFF. Total runtime: ~3 minutes.
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
print("RSNA Knee: Full End-to-End Pipeline (Train & Submit)")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Execution Device: {DEVICE}")

# -------------------------------------------------------------------------
# 1. PATH RESOLUTION
# -------------------------------------------------------------------------
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

if comp_dir is None:
    raise FileNotFoundError("Could not locate competition files in /kaggle/input")

print(f"Competition Root: {comp_dir}")

train_df = pd.read_csv(os.path.join(comp_dir, 'train.csv'))
train_series_df = pd.read_csv(os.path.join(comp_dir, 'train_series.csv'))
test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
target_cols = [c for c in sample_sub.columns if c != 'StudyInstanceUID']

PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

# -------------------------------------------------------------------------
# 2. SERIES SELECTION MAPPER (1 Best Series per Plane)
# -------------------------------------------------------------------------
def build_series_map(ts_df):
    sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
    ascending = [True, True]
    if 'Fluid_Sensitive' in ts_df.columns:
        sort_cols.append('Fluid_Sensitive')
        ascending.append(False)
    if 'Fat_Suppression' in ts_df.columns:
        sort_cols.append('Fat_Suppression')
        ascending.append(False)
    
    selected = ts_df.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])
    s_map = {}
    for _, row in selected.iterrows():
        study = row['StudyInstanceUID']
        plane = row['Anatomical_Plane']
        series = row['SeriesInstanceUID']
        if study not in s_map:
            s_map[study] = {}
        s_map[study][plane] = series
    return s_map

train_series_map = build_series_map(train_series_df)
test_series_map = build_series_map(test_series_df)

# -------------------------------------------------------------------------
# 3. FAST 2.5D SLICE EXTRACTOR
# -------------------------------------------------------------------------
def load_volume(base_folder, study_id, plane, s_map, img_size=224):
    if study_id not in s_map or plane not in s_map[study_id]:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    
    series_id = s_map[study_id][plane]
    series_path = os.path.join(comp_dir, base_folder, str(study_id), str(series_id))
    if not os.path.exists(series_path):
        series_path = os.path.join(comp_dir, base_folder, str(series_id))
    if not os.path.exists(series_path):
        return np.zeros((3, img_size, img_size), dtype=np.float32)

    files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')])
    if len(files) == 0:
        return np.zeros((3, img_size, img_size), dtype=np.float32)

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

# -------------------------------------------------------------------------
# 4. PRE-CACHE 58 TRAINING STUDIES IN RAM (~10s)
# -------------------------------------------------------------------------
labeled_df = train_df[train_df['ACL'].notna()].copy().reset_index(drop=True)
print(f"\n[CACHE] Pre-caching {len(labeled_df)} annotated studies into RAM...")
all_sag, all_cor, all_axi, all_labels = [], [], [], []

for _, row in labeled_df.iterrows():
    study_id = row['StudyInstanceUID']
    all_sag.append(load_volume('train_series', study_id, 'Sagittal', train_series_map))
    all_cor.append(load_volume('train_series', study_id, 'Coronal', train_series_map))
    all_axi.append(load_volume('train_series', study_id, 'Axial', train_series_map))
    all_labels.append(row[target_cols].values.astype(np.float32))

sag_tensor = torch.tensor(np.array(all_sag), dtype=torch.float32)
cor_tensor = torch.tensor(np.array(all_cor), dtype=torch.float32)
axi_tensor = torch.tensor(np.array(all_axi), dtype=torch.float32)
labels_tensor = torch.tensor(np.array(all_labels), dtype=torch.float32)

# -------------------------------------------------------------------------
# 5. MULTI-PLANE NEURAL NETWORK
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 6. MODEL TRAINING (15 Epochs on GPU ~35s)
# -------------------------------------------------------------------------
full_dataset = TensorDataset(sag_tensor, cor_tensor, axi_tensor, labels_tensor)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(labeled_df))

train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=8, shuffle=True)
val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=8, shuffle=False)

model = MultiPlaneKneeNet(num_classes=12).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

print(f"\n[TRAINING] Training Multi-Plane ResNet on {len(train_idx)} cases...")
best_auc = 0.0
model_save_path = '/kaggle/working/best_multiplane_knee_model.pth'

for epoch in range(1, 16):
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
    print(f"Epoch {epoch:2d}/15 | Val Macro ROC-AUC: {mean_auc:.4f}")

    if mean_auc > best_auc:
        best_auc = mean_auc
        torch.save(model.state_dict(), model_save_path)

print(f"\n[TRAINING COMPLETE] Best Val AUC: {best_auc:.4f}")

# -------------------------------------------------------------------------
# 7. INFERENCE ON TEST SET & GENERATE SUBMISSION
# -------------------------------------------------------------------------
print(f"\n[INFERENCE] Predicting across {len(test_df)} test studies...")
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
model.eval()

predictions = []
for i, study_id in enumerate(test_df['StudyInstanceUID']):
    sag = torch.tensor(load_volume('test_series', study_id, 'Sagittal', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    cor = torch.tensor(load_volume('test_series', study_id, 'Coronal', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    axi = torch.tensor(load_volume('test_series', study_id, 'Axial', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        raw_pred = torch.sigmoid(model(sag, cor, axi)).cpu().numpy()[0]
        blended = [0.85 * raw_pred[j] + 0.15 * PRIORS[col] for j, col in enumerate(target_cols)]
        predictions.append(blended)

sub = pd.DataFrame(predictions, columns=target_cols)
sub.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
sub.to_csv('/kaggle/working/submission.csv', index=False)

print("=" * 70)
print("[SUCCESS] submission.csv created successfully!")
print(sub.head(3))
print("=" * 70)
