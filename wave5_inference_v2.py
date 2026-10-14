"""
RSNA Knee Abnormality Detection — Improved Wave 5 v2
Offline Test Inference with v2 model checkpoint.
"""
import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("RSNA Knee: Wave 5 v2 — Offline Inference")
print("=" * 70)

# 1. Path Resolution
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
target_cols = [c for c in sample_sub.columns if c != 'StudyInstanceUID']

PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

# 2. Series Map
def build_series_map(ts_df):
    sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
    ascending = [True, True]
    if 'Fluid_Sensitive' in ts_df.columns:
        sort_cols.append('Fluid_Sensitive'); ascending.append(False)
    if 'Fat_Suppression' in ts_df.columns:
        sort_cols.append('Fat_Suppression'); ascending.append(False)
    selected = ts_df.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])
    s_map = {}
    for _, row in selected.iterrows():
        study = row['StudyInstanceUID']
        plane = row['Anatomical_Plane']
        if study not in s_map:
            s_map[study] = {}
        s_map[study][plane] = row['SeriesInstanceUID']
    return s_map

test_series_map = build_series_map(test_series_df)

# 3. Slice Loader
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
    if not files:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    mid = len(files) // 2
    indices = [max(0, mid - 2), mid, min(len(files) - 1, mid + 2)]
    channels = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx])
            img = dcm.pixel_array.astype(np.float32)
            mn, mx = img.min(), img.max()
            img = (img - mn) / (mx - mn + 1e-8)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            img = np.zeros((img_size, img_size), dtype=np.float32)
        channels.append(img)
    return np.stack(channels, axis=0)

# 4. Model
class MultiPlaneKneeNetV2(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        base = models.resnet34(weights=None)
        feature_dim = 512
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feature_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, sag, cor, axi):
        f = lambda x: self.backbone(x).flatten(1)
        fused = torch.cat([f(sag), f(cor), f(axi)], dim=1)
        return self.classifier(fused)

model = MultiPlaneKneeNetV2().to(DEVICE)
v2_path = '/kaggle/working/best_knee_v2.pth'
v1_path = '/kaggle/working/best_multiplane_knee_model.pth'

if os.path.exists(v2_path):
    model.load_state_dict(torch.load(v2_path, map_location=DEVICE))
    print(f"Loaded v2 model from: {v2_path}")
elif os.path.exists(v1_path):
    print(f"[FALLBACK] Loading v1 model. Run wave4_train_v2.py for better results.")
    # Load with v1 architecture
    class MultiPlaneKneeNetV1(nn.Module):
        def __init__(self, num_classes=12):
            super().__init__()
            base = models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(512*3, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_classes))
        def forward(self, sag, cor, axi):
            f = lambda x: self.backbone(x).flatten(1)
            return self.classifier(torch.cat([f(sag), f(cor), f(axi)], dim=1))
    model = MultiPlaneKneeNetV1().to(DEVICE)
    model.load_state_dict(torch.load(v1_path, map_location=DEVICE))
else:
    print("[WARNING] No weights found! Using class priors only.")
    model = None

if model:
    model.eval()

# 5. Inference
print(f"\n[INFERENCE] Predicting {len(test_df)} test studies...")
predictions = []
for i, study_id in enumerate(test_df['StudyInstanceUID']):
    sag = torch.tensor(load_volume('test_series', study_id, 'Sagittal', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    cor = torch.tensor(load_volume('test_series', study_id, 'Coronal', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    axi = torch.tensor(load_volume('test_series', study_id, 'Axial', test_series_map), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    if model is not None:
        with torch.no_grad():
            raw = torch.sigmoid(model(sag, cor, axi)).cpu().numpy()[0]
        blended = [0.85 * raw[j] + 0.15 * PRIORS[col] for j, col in enumerate(target_cols)]
    else:
        blended = [PRIORS[col] for col in target_cols]
    predictions.append(blended)

sub = pd.DataFrame(predictions, columns=target_cols)
sub.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
sub.to_csv('/kaggle/working/submission.csv', index=False)
print(f"\n[SUCCESS] submission.csv generated ({sub.shape})")
print(sub.head(3))
print("=" * 70)
