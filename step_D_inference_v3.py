"""
RSNA Knee — Improvement Module D: Offline Inference v3 (with TTA)
Loads best_knee_v3.pth and predicts on test set with:
  - 7-slice equidistant sampling
  - CLAHE preprocessing
  - Test-Time Augmentation (horizontal flip average)
  - Class prior blending for calibration
"""
import os
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    HAS_EFFNET = True
except ImportError:
    HAS_EFFNET = False
    from torchvision import models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_SLICES = 7
IMG_SIZE = 224
TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

print("=" * 70)
print("RSNA Knee: Step D — Offline Inference v3 (EfficientNet + TTA)")
print("=" * 70)

# 1. Path
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
print(f"Test studies: {len(test_df)}")

# 2. Series map
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

test_map = build_series_map(test_series_df)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 3. Slice loader
def load_test_volume(study_id, plane, n=N_SLICES, sz=IMG_SIZE):
    if study_id not in test_map or plane not in test_map[study_id]:
        return np.zeros((n, sz, sz), dtype=np.float32)
    series_id = test_map[study_id][plane]
    path = os.path.join(comp_dir, 'test_series', str(study_id), str(series_id))
    if not os.path.exists(path):
        path = os.path.join(comp_dir, 'test_series', str(series_id))
    if not os.path.exists(path):
        return np.zeros((n, sz, sz), dtype=np.float32)
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')])
    if not files:
        return np.zeros((n, sz, sz), dtype=np.float32)
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    channels = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx])
            img = dcm.pixel_array.astype(np.float32)
            mn, mx = img.min(), img.max()
            u8 = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
            img_c = clahe.apply(u8).astype(np.float32) / 255.0
            img_r = cv2.resize(img_c, (sz, sz), interpolation=cv2.INTER_AREA)
        except Exception:
            img_r = np.zeros((sz, sz), dtype=np.float32)
        channels.append(img_r)
    return np.stack(channels, axis=0)

# 4. Model definition (must match training architecture)
class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        if HAS_EFFNET:
            base = efficientnet_b3(weights=None)
            old_conv = base.features[0][0]
            base.features[0][0] = nn.Conv2d(1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=False)
            self.backbone = base.features
            self.pool = base.avgpool
            self.feat_dim = 1536
        else:
            base = models.resnet34(weights=None)
            base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.feat_dim = 512

    def forward(self, x):
        B, S, H, W = x.shape
        x = x.view(B * S, 1, H, W)
        f = self.pool(self.backbone(x))
        f = f.view(B, S, self.feat_dim)
        return f.mean(dim=1)

class MultiPlaneKneeNetV3(nn.Module):
    def __init__(self, nc=12):
        super().__init__()
        self.sag_enc = PlaneEncoder()
        self.cor_enc = PlaneEncoder()
        self.axi_enc = PlaneEncoder()
        fd = self.sag_enc.feat_dim
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(fd*3, 512),
            nn.LayerNorm(512), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(512, nc)
        )

    def forward(self, sag, cor, axi):
        fused = torch.cat([self.sag_enc(sag), self.cor_enc(cor), self.axi_enc(axi)], dim=1)
        return self.head(fused)

# 5. Load model
model = MultiPlaneKneeNetV3().to(DEVICE)
v3_path = '/kaggle/working/best_knee_v3.pth'
v2_path = '/kaggle/working/best_knee_v2.pth'
v1_path = '/kaggle/working/best_multiplane_knee_model.pth'

if os.path.exists(v3_path):
    model.load_state_dict(torch.load(v3_path, map_location=DEVICE))
    print(f"Loaded v3 model: {v3_path}")
else:
    raise FileNotFoundError("best_knee_v3.pth not found! Run step_C_train_v3.py first.")

model.eval()

# 6. Predict with TTA
print(f"\n[INFERENCE + TTA] Processing {len(test_df)} studies...")
predictions = []

for i, study_id in enumerate(test_df['StudyInstanceUID']):
    if (i + 1) % 200 == 0 or (i + 1) == len(test_df):
        print(f"  {i+1}/{len(test_df)}")

    sag_np = load_test_volume(study_id, 'Sagittal')
    cor_np = load_test_volume(study_id, 'Coronal')
    axi_np = load_test_volume(study_id, 'Axial')

    def to_t(arr):
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    sag_t, cor_t, axi_t = to_t(sag_np), to_t(cor_np), to_t(axi_np)

    with torch.no_grad():
        p1 = torch.sigmoid(model(sag_t, cor_t, axi_t)).cpu().numpy()[0]
        p2 = torch.sigmoid(model(
            sag_t.flip(-1), cor_t.flip(-1), axi_t.flip(-1)
        )).cpu().numpy()[0]
        raw = (p1 + p2) / 2

    # Blend: 90% model + 10% prior
    blended = [0.90 * raw[j] + 0.10 * PRIORS[col] for j, col in enumerate(TARGET_COLS)]
    predictions.append(blended)

sub = pd.DataFrame(predictions, columns=TARGET_COLS)
sub.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])
sub.to_csv('/kaggle/working/submission.csv', index=False)

print("\n" + "=" * 70)
print(f"[SUCCESS] submission.csv saved! Shape: {sub.shape}")
print(sub.head(3))
print("=" * 70)
