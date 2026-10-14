"""
RSNA Knee — Step D v2: GPU-Accelerated Batched Offline Inference
OPTIMIZATIONS:
  1. ThreadPoolExecutor: Parallel DICOM disk reads (4 CPU workers)
  2. GPU Tensor Processing: F.interpolate bilinear resize + normalization on GPU
  3. Batched GPU Inference: processes test cases in batches through PyTorch on GPU
  4. Mixed Precision (torch.cuda.amp.autocast): fast FP16 inference on T4 Tensor Cores
  5. Test-Time Augmentation (TTA): original + horizontal flip average
  6. Exact matching architecture with step_C_train_v3 (cross-plane attention gate)
  7. Bayesian prior blending for robust calibration
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

print("=" * 70)
print("RSNA Knee: Step D — GPU-Accelerated Batched Inference (AMP + TTA)")
print("=" * 70)

# -----------------------------------------------------------------------
# 1. Device Setup
# -----------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Inference Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

N_SLICES = 7
IMG_SIZE = 224
INFER_BATCH_SIZE = 8  # studies batched together on GPU

TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

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
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
print(f"Loaded test studies to predict: {len(test_df)}")

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
    sel = (ts_df.sort_values(sort_cols, ascending=asc)
           .drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane']))
    m = {}
    for _, row in sel.iterrows():
        s = row['StudyInstanceUID']
        if s not in m:
            m[s] = {}
        m[s][row['Anatomical_Plane']] = row['SeriesInstanceUID']
    return m

test_series_map = build_series_map(test_series_df)

# -----------------------------------------------------------------------
# 4. Fast Parallel DICOM Slice Reader
# -----------------------------------------------------------------------
def read_test_series_raw(study_id, plane):
    zeros = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if study_id not in test_series_map or plane not in test_series_map[study_id]:
        return zeros

    series_id = test_series_map[study_id][plane]
    path = os.path.join(comp_dir, 'test_series', str(study_id), str(series_id))
    if not os.path.exists(path):
        path = os.path.join(comp_dir, 'test_series', str(series_id))
    if not os.path.exists(path):
        return zeros

    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')])
    if not files:
        return zeros

    indices = np.linspace(0, len(files) - 1, N_SLICES, dtype=int)
    slices = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx], stop_before_pixels=False)
            arr = dcm.pixel_array.astype(np.float32)
            if arr.shape != (IMG_SIZE, IMG_SIZE):
                arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        except Exception:
            arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        slices.append(arr)
    return slices

def read_test_study_raw(study_id):
    sag = read_test_series_raw(study_id, 'Sagittal')
    cor = read_test_series_raw(study_id, 'Coronal')
    axi = read_test_series_raw(study_id, 'Axial')
    return study_id, sag, cor, axi

# -----------------------------------------------------------------------
# 5. GPU Preprocessing (Identical to Step B for domain consistency)
# -----------------------------------------------------------------------
def preprocess_slices_gpu(slices_list, device=DEVICE, target_size=IMG_SIZE):
    if isinstance(slices_list, np.ndarray):
        return slices_list

    tensors = []
    for arr in slices_list:
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            tensors.append(torch.zeros(1, 1, target_size, target_size))
        else:
            t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensors.append(t)

    batch_t = torch.cat(tensors, dim=0).to(device)  # (N_SLICES, 1, H, W)
    batch_resized = F.interpolate(batch_t, size=(target_size, target_size),
                                  mode='bilinear', align_corners=False)

    B = batch_resized.shape[0]
    flat = batch_resized.view(B, -1)
    mins = flat.min(dim=1).values.view(B, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(B, 1, 1, 1)
    batch_norm = (batch_resized - mins) / (maxs - mins + 1e-8)

    block_size = target_size // 4
    for bx in range(4):
        for by in range(4):
            xs, xe = bx * block_size, (bx + 1) * block_size
            ys, ye = by * block_size, (by + 1) * block_size
            block = batch_norm[:, :, xs:xe, ys:ye]
            bflat = block.reshape(B, -1)
            lo = torch.quantile(bflat, 0.02, dim=1).view(B, 1, 1, 1)
            hi = torch.quantile(bflat, 0.98, dim=1).view(B, 1, 1, 1)
            batch_norm[:, :, xs:xe, ys:ye] = torch.clamp((block - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    return batch_norm.squeeze(1)  # (N_SLICES, H, W) on GPU tensor

# -----------------------------------------------------------------------
# 6. Model Definition (Exact Match with Step C)
# -----------------------------------------------------------------------
from torchvision import models

class PlaneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        loaded = False
        try:
            from torchvision.models import efficientnet_b3
            base = efficientnet_b3(weights=None)
            old_conv = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            self.backbone = base.features
            self.pool = base.avgpool
            self.feat_dim = 1536
            loaded = True
        except Exception:
            pass

        if not loaded:
            base = models.resnet34(weights=None)
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
        feat_dim = self.sag_enc.feat_dim

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
# 7. Checkpoint Loading
# -----------------------------------------------------------------------
model = MultiPlaneKneeNetV3(num_classes=12).to(DEVICE)
v3_path = '/kaggle/working/best_knee_v3.pth'
v2_path = '/kaggle/working/best_knee_v2.pth'
v1_path = '/kaggle/working/best_multiplane_knee_model.pth'

if os.path.exists(v3_path):
    state_dict = torch.load(v3_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"[MODEL LOADED] Successfully loaded v3 weights: {v3_path}")
elif os.path.exists(v2_path):
    print(f"[MODEL WARNING] v3 not found. Loading v2 weights: {v2_path}")
    state_dict = torch.load(v2_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
elif os.path.exists(v1_path):
    print(f"[MODEL WARNING] Loading v1 weights: {v1_path}")
    state_dict = torch.load(v1_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
else:
    print("[MODEL ALERT] No model weights detected! Prior-only fallback active.")
    model = None

if model is not None:
    model.eval()

# -----------------------------------------------------------------------
# 8. Parallel Loading + Batched GPU Inference with TTA
# -----------------------------------------------------------------------
test_uids = test_df['StudyInstanceUID'].tolist()
print(f"\n[INFERENCE] Launching parallel loader + GPU batched inference for {len(test_uids)} studies...")

study_raw_dict = {}
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(read_test_study_raw, sid): sid for sid in test_uids}
    for future in futures:
        sid, sag_raw, cor_raw, axi_raw = future.result()
        study_raw_dict[sid] = (sag_raw, cor_raw, axi_raw)

predictions_dict = {}

# Process in GPU batches
for i in range(0, len(test_uids), INFER_BATCH_SIZE):
    batch_uids = test_uids[i: i + INFER_BATCH_SIZE]
    batch_sag, batch_cor, batch_axi = [], [], []

    for sid in batch_uids:
        sag_raw, cor_raw, axi_raw = study_raw_dict[sid]
        batch_sag.append(preprocess_slices_gpu(sag_raw))
        batch_cor.append(preprocess_slices_gpu(cor_raw))
        batch_axi.append(preprocess_slices_gpu(axi_raw))

    sag_tensor = torch.stack(batch_sag, dim=0)  # (B, S, H, W)
    cor_tensor = torch.stack(batch_cor, dim=0)
    axi_tensor = torch.stack(batch_axi, dim=0)

    if model is not None:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                # Standard pass
                p1 = torch.sigmoid(model(sag_tensor, cor_tensor, axi_tensor))
                # TTA horizontal flip pass
                p2 = torch.sigmoid(model(sag_tensor.flip(-1), cor_tensor.flip(-1), axi_tensor.flip(-1)))
                p_avg = ((p1 + p2) * 0.5).cpu().numpy()

        for idx, sid in enumerate(batch_uids):
            raw = p_avg[idx]
            # 90% neural network prediction + 10% competition class prior
            blended = [0.90 * raw[j] + 0.10 * PRIORS[col] for j, col in enumerate(TARGET_COLS)]
            predictions_dict[sid] = blended
    else:
        for sid in batch_uids:
            predictions_dict[sid] = [PRIORS[col] for col in TARGET_COLS]

    if (i + len(batch_uids)) % 100 == 0 or (i + len(batch_uids)) == len(test_uids):
        gpu_mem = f" | VRAM: {torch.cuda.memory_allocated() / 1e6:.0f}MB" if DEVICE.type == 'cuda' else ""
        print(f"  Processed {i + len(batch_uids)}/{len(test_uids)} studies{gpu_mem}")

# -----------------------------------------------------------------------
# 9. Format & Save Final Submission File
# -----------------------------------------------------------------------
final_rows = [predictions_dict[sid] for sid in test_uids]
sub = pd.DataFrame(final_rows, columns=TARGET_COLS)
sub.insert(0, 'StudyInstanceUID', test_uids)

sub_path = '/kaggle/working/submission.csv'
sub.to_csv(sub_path, index=False)

print("\n" + "=" * 70)
print(f"[SUCCESS] Final submission verified: {sub_path}")
print(f"Shape: {sub.shape} | Null Count: {sub.isnull().sum().sum()}")
print("\nPreview of first 3 predictions:")
print(sub.head(3).to_string())
print("=" * 70)
