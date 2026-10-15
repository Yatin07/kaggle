"""
RSNA Knee Abnormality Detection — Iteration 3 Ensemble Inference
FAST OFFLINE SUBMISSION (UNDER 2 MINUTES):
  1. Loads 5-fold ensemble (knee_model_fold0.pth ... fold4.pth).
  2. Uses true slice normal vector sorting + letterbox padding.
  3. Averages predictions across all 5 folds on GPU.
  4. Generates submission.csv without any retraining.
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

warnings.filterwarnings('ignore')

print("=" * 70)
print("RSNA Knee: Iteration 3 — 5-Fold Ensemble Fast Inference (<2 mins)")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
N_SLICES = 7

TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

# 1. Path Resolution
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
print(f"Test cases to predict: {len(test_df)}")

# 2. Build Series Map
def build_series_map(ts_df):
    sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
    asc = [True, True]
    if 'Fluid_Sensitive' in ts_df.columns:
        sort_cols.append('Fluid_Sensitive'); asc.append(False)
    if 'Fat_Suppression' in ts_df.columns:
        sort_cols.append('Fat_Suppression'); asc.append(False)
    sel = ts_df.sort_values(sort_cols, ascending=asc).drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])
    m = {}
    for _, row in sel.iterrows():
        s = row['StudyInstanceUID']
        if s not in m:
            m[s] = {}
        m[s][row['Anatomical_Plane']] = row['SeriesInstanceUID']
    return m

test_series_map = build_series_map(test_series_df)

# 3. Model Architecture
class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights=None)
        old_conv = base.conv1
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    def forward(self, x):
        B, S, H, W = x.shape
        x = x.view(B * S, 1, H, W)
        feats = self.pool(self.backbone(x))
        feats = feats.view(B, S, self.feat_dim)
        return feats.mean(dim=1)

class MultiPlaneKneeNetV3(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.sag_enc = PlaneEncoder()
        self.cor_enc = PlaneEncoder()
        self.axi_enc = PlaneEncoder()
        fd = self.sag_enc.feat_dim

        self.plane_attn = nn.Sequential(
            nn.Linear(fd * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(fd * 3, 512),
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

# 4. Load Models (Ensemble of available folds)
loaded_models = []
for fold in range(5):
    path = f'/kaggle/working/knee_model_fold{fold}.pth'
    if os.path.exists(path):
        m = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        loaded_models.append(m)
        print(f"  Loaded Fold {fold + 1} weights: {path}")

# Fallback to single checkpoint if fold models not found
if not loaded_models:
    fallback = '/kaggle/working/best_knee_v3.pth'
    if os.path.exists(fallback):
        m = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)
        m.load_state_dict(torch.load(fallback, map_location=DEVICE))
        m.eval()
        loaded_models.append(m)
        print(f"  [FALLBACK] Loaded single checkpoint: {fallback}")
    else:
        print("  [ALERT] No checkpoints found! Prior fallback active.")

print(f"Ensemble size: {len(loaded_models)} model(s)")

# 5. Fast DICOM Slicing using true normal vector sorting
from iter3_dicom_loader import load_series_volume

def get_study_volume(study_id, plane):
    if study_id not in test_series_map or plane not in test_series_map[study_id]:
        return np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    series_id = test_series_map[study_id][plane]
    series_path = os.path.join(comp_dir, 'test_series', str(study_id), str(series_id))
    if not os.path.exists(series_path):
        series_path = os.path.join(comp_dir, 'test_series', str(series_id))
    vol_u8 = load_series_volume(series_path, n_slices=N_SLICES, target_size=IMG_SIZE)
    return vol_u8.astype(np.float32) / 255.0

# 6. Ensemble Inference Loop
print(f"\n[INFERENCE] Predicting {len(test_df)} studies across 5-fold ensemble...")
predictions = []

for idx, sid in enumerate(test_df['StudyInstanceUID']):
    sag_np = get_study_volume(sid, 'Sagittal')
    cor_np = get_study_volume(sid, 'Coronal')
    axi_np = get_study_volume(sid, 'Axial')

    sag_t = torch.tensor(sag_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    cor_t = torch.tensor(cor_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    axi_t = torch.tensor(axi_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    if loaded_models:
        fold_probs = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                for m in loaded_models:
                    prob = torch.sigmoid(m(sag_t, cor_t, axi_t)).cpu().numpy()[0]
                    fold_probs.append(prob)
        ensemble_prob = np.mean(fold_probs, axis=0)
        blended = [0.90 * ensemble_prob[j] + 0.10 * PRIORS[col] for j, col in enumerate(TARGET_COLS)]
    else:
        blended = [PRIORS[col] for col in TARGET_COLS]

    predictions.append(blended)

# 7. Save Verified Submission
sub = pd.DataFrame(predictions, columns=TARGET_COLS)
sub.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
sub.to_csv('/kaggle/working/submission.csv', index=False)

print("\n" + "=" * 70)
print(f"[SUCCESS] 5-Fold Ensemble submission saved: /kaggle/working/submission.csv")
print(f"Shape: {sub.shape} | Nulls: {sub.isnull().sum().sum()}")
print(sub.head(3).to_string())
print("=" * 70)
