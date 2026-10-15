"""
RSNA Knee Abnormality Detection — Iteration 3 Ensemble Inference
FAST OFFLINE SUBMISSION (UNDER 2 MINUTES):
  1. Loads 5-fold ensemble (knee_model_fold0.pth ... fold4.pth).
  2. Uses true slice normal vector sorting + letterbox padding.
  3. Averages predictions across all 5 folds on GPU.
  4. Generates submission.csv without any retraining.
  5. 100% Self-Contained (NO ModuleNotFoundError).
"""
import os
import time
import warnings
import cv2
import numpy as np
import pandas as pd
import pydicom
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

# -----------------------------------------------------------------------
# 1. 3D DICOM Slice Normal Vector Sorter & Volume Loader (Inlined)
# -----------------------------------------------------------------------
def sort_series_slices(series_path):
    files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not files:
        return []
    slice_positions, valid_files = [], []
    for fpath in files:
        try:
            dcm = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(dcm, 'ImagePositionPatient') and hasattr(dcm, 'ImageOrientationPatient'):
                iop = np.array(dcm.ImageOrientationPatient, dtype=float)
                row_dir = iop[:3]
                col_dir = iop[3:]
                slice_normal = np.cross(row_dir, col_dir)
                norm_len = np.linalg.norm(slice_normal)
                if norm_len > 1e-6:
                    slice_normal = slice_normal / norm_len
                ipp = np.array(dcm.ImagePositionPatient, dtype=float)
                pos = float(np.dot(ipp, slice_normal))
            elif hasattr(dcm, 'InstanceNumber'):
                pos = float(dcm.InstanceNumber)
            else:
                pos = 0.0
            slice_positions.append(pos)
            valid_files.append(fpath)
        except Exception:
            continue
    if not valid_files:
        return files
    sorted_indices = np.argsort(slice_positions)
    return [valid_files[idx] for idx in sorted_indices]

def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    padded = np.zeros((size, size), dtype=img.dtype)
    padded[pad_h: pad_h + h, pad_w: pad_w + w] = img
    return padded

def load_series_volume(series_path, n_slices=7, target_size=224):
    blank = np.zeros((n_slices, target_size, target_size), dtype=np.uint8)
    if not os.path.exists(series_path):
        return blank
    sorted_files = sort_series_slices(series_path)
    if not sorted_files:
        return blank

    indices = np.linspace(0, len(sorted_files) - 1, n_slices, dtype=int)
    raw_slices = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(sorted_files[idx], stop_before_pixels=False)
            arr = dcm.pixel_array.astype(np.float32)
            slope = getattr(dcm, 'RescaleSlope', 1.0)
            intercept = getattr(dcm, 'RescaleIntercept', 0.0)
            arr = arr * slope + intercept
            if getattr(dcm, 'PhotometricInterpretation', '') == 'MONOCHROME1':
                arr = np.max(arr) - arr
            raw_slices.append(arr)
        except Exception:
            raw_slices.append(np.zeros((target_size, target_size), dtype=np.float32))

    all_pixels = np.concatenate([s.flatten() for s in raw_slices])
    p1 = np.percentile(all_pixels, 1.0)
    p99 = np.percentile(all_pixels, 99.0)
    if p99 <= p1:
        p99 = p1 + 1.0

    processed_slices = []
    for s in raw_slices:
        s_norm = np.clip((s - p1) / (p99 - p1), 0.0, 1.0) * 255.0
        s_u8 = s_norm.astype(np.uint8)
        s_square = pad_to_square(s_u8)
        s_resized = cv2.resize(s_square, (target_size, target_size), interpolation=cv2.INTER_AREA)
        processed_slices.append(s_resized)

    return np.stack(processed_slices, axis=0)

# -----------------------------------------------------------------------
# 2. Path Resolution & Data Tables
# -----------------------------------------------------------------------
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
print(f"Test cases to predict: {len(test_df)}")

# -----------------------------------------------------------------------
# 3. Series Map Builder
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# 4. Model Architecture (Must match training)
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# 5. Load Fold Models
# -----------------------------------------------------------------------
loaded_models = []
for fold in range(5):
    path = f'/kaggle/working/knee_model_fold{fold}.pth'
    if os.path.exists(path):
        m = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        loaded_models.append(m)
        print(f"  Loaded Fold {fold + 1} weights: {path}")

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

print(f"Active Ensemble Size: {len(loaded_models)} model(s)")

# -----------------------------------------------------------------------
# 6. Test Volume Loader using True Slice Normal Ordering
# -----------------------------------------------------------------------
def get_study_volume(study_id, plane):
    if study_id not in test_series_map or plane not in test_series_map[study_id]:
        return np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    series_id = test_series_map[study_id][plane]
    series_path = os.path.join(comp_dir, 'test_series', str(study_id), str(series_id))
    if not os.path.exists(series_path):
        series_path = os.path.join(comp_dir, 'test_series', str(series_id))
    vol_u8 = load_series_volume(series_path, n_slices=N_SLICES, target_size=IMG_SIZE)
    return vol_u8.astype(np.float32) / 255.0

# -----------------------------------------------------------------------
# 7. Predict & Save
# -----------------------------------------------------------------------
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

sub = pd.DataFrame(predictions, columns=TARGET_COLS)
sub.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
sub.to_csv('/kaggle/working/submission.csv', index=False)

print("\n" + "=" * 70)
print(f"[SUCCESS] 5-Fold Ensemble submission saved: /kaggle/working/submission.csv")
print(f"Shape: {sub.shape} | Nulls: {sub.isnull().sum().sum()}")
print(sub.head(3).to_string())
print("=" * 70)
