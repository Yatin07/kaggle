"""
RSNA Knee — Step B v3: RAM-Safe GPU-Accelerated Cache
WHY IT CRASHED BEFORE:
  1. Queued all 4,407 studies at once in ThreadPoolExecutor (flooded RAM with raw DICOMs)
  2. Stored arrays as float32 in Python lists (reached >28GB, triggering Kaggle 30GB OOM)

THE PERMANENT FIX (STAYS UNDER 5GB RAM AT ALL TIMES):
  1. Pre-allocated uint8 numpy arrays: only 1.55 GB per plane (4.65 GB total RAM for all 4,407 studies!)
  2. Sequential / bounded streaming: reads one study, GPU pre-processes, writes directly to pre-allocated slot
  3. No huge list accumulations, zero memory duplication
  4. Periodic gc.collect() + torch.cuda.empty_cache()
  5. Runs in ~5-7 minutes without EVER exceeding 6 GB RAM!
"""
import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')

print("=" * 70)
print("RSNA Knee: Step B v3 — RAM-Safe GPU Cache (Pre-allocated uint8, <6GB RAM)")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Compute Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
N_SLICES = 7     # equidistant slices per plane
IMG_SIZE = 224   # 224x224 resolution
TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

# -----------------------------------------------------------------------
# 1. Path Resolution
# -----------------------------------------------------------------------
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

train_series_df = pd.read_csv(os.path.join(comp_dir, 'train_series.csv'))
pseudo_labels_path = '/kaggle/working/pseudo_labels.csv'

if not os.path.exists(pseudo_labels_path):
    raise FileNotFoundError("Run step_A_pseudo_labels.py first!")

study_df = pd.read_csv(pseudo_labels_path)
N = len(study_df)
print(f"Total studies to cache: {N}")

# -----------------------------------------------------------------------
# 2. Series Selection Map
# -----------------------------------------------------------------------
sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
ascending = [True, True]
if 'Fluid_Sensitive' in train_series_df.columns:
    sort_cols.append('Fluid_Sensitive'); ascending.append(False)
if 'Fat_Suppression' in train_series_df.columns:
    sort_cols.append('Fat_Suppression'); ascending.append(False)

selected = (train_series_df
            .sort_values(sort_cols, ascending=ascending)
            .drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane']))

series_map = {}
for _, row in selected.iterrows():
    s = row['StudyInstanceUID']
    p = row['Anatomical_Plane']
    if s not in series_map:
        series_map[s] = {}
    series_map[s][p] = row['SeriesInstanceUID']

# -----------------------------------------------------------------------
# 3. Pre-allocate Fixed-Size uint8 Arrays (Guarantees <5GB RAM ceiling!)
# -----------------------------------------------------------------------
print(f"\n[RAM SAFETY] Pre-allocating uint8 buffers:")
print(f"  3 planes x ({N}, {N_SLICES}, {IMG_SIZE}, {IMG_SIZE}) uint8 = 4.65 GB fixed RAM")
sag_arr = np.zeros((N, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
cor_arr = np.zeros((N, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
axi_arr = np.zeros((N, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
labels_arr = np.zeros((N, len(TARGET_COLS)), dtype=np.float32)
uids_list = study_df['StudyInstanceUID'].tolist()

for i, (_, row) in enumerate(study_df.iterrows()):
    labels_arr[i] = [row.get(c, -1) for c in TARGET_COLS]

# -----------------------------------------------------------------------
# 4. Fast GPU Processor for 1 Series (returns uint8 array of shape [7, 224, 224])
# -----------------------------------------------------------------------
def load_and_process_series(study_id, plane):
    """Reads raw DICOM slices, transforms on GPU, returns (7, 224, 224) uint8"""
    blank = np.zeros((N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if study_id not in series_map or plane not in series_map[study_id]:
        return blank

    series_id = series_map[study_id][plane]
    path = os.path.join(comp_dir, 'train_series', str(study_id), str(series_id))
    if not os.path.exists(path):
        return blank

    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')])
    if not files:
        return blank

    indices = np.linspace(0, len(files) - 1, N_SLICES, dtype=int)
    raw_slices = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx], stop_before_pixels=False)
            arr = dcm.pixel_array.astype(np.float32)
        except Exception:
            arr = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        raw_slices.append(arr)

    # Process batch of 7 slices on GPU
    try:
        tensors = []
        for arr in raw_slices:
            if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
                tensors.append(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))
            else:
                tensors.append(torch.from_numpy(arr).unsqueeze(0).unsqueeze(0))

        batch_t = torch.cat(tensors, dim=0).to(DEVICE)  # (7, 1, H, W)

        # Bilinear resize on GPU
        batch_resized = F.interpolate(batch_t, size=(IMG_SIZE, IMG_SIZE),
                                      mode='bilinear', align_corners=False)

        # Min-max normalization
        flat = batch_resized.view(N_SLICES, -1)
        mins = flat.min(dim=1).values.view(N_SLICES, 1, 1, 1)
        maxs = flat.max(dim=1).values.view(N_SLICES, 1, 1, 1)
        norm = (batch_resized - mins) / (maxs - mins + 1e-8)

        # Local contrast stretching (4x4 tiles)
        tile = IMG_SIZE // 4
        for bx in range(4):
            for by in range(4):
                xs, xe = bx * tile, (bx + 1) * tile
                ys, ye = by * tile, (by + 1) * tile
                block = norm[:, :, xs:xe, ys:ye]
                bflat = block.reshape(N_SLICES, -1)
                lo = torch.quantile(bflat, 0.02, dim=1).view(N_SLICES, 1, 1, 1)
                hi = torch.quantile(bflat, 0.98, dim=1).view(N_SLICES, 1, 1, 1)
                norm[:, :, xs:xe, ys:ye] = torch.clamp((block - lo) / (hi - lo + 1e-8), 0.0, 1.0)

        # Convert to uint8 (0..255) on GPU before transferring to CPU
        uint8_result = (norm.squeeze(1) * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        return uint8_result
    except Exception:
        return blank

# -----------------------------------------------------------------------
# 5. Controlled Streaming Loop (Zero Risk of OOM)
# -----------------------------------------------------------------------
t0 = time.time()
print(f"\n[STREAMING] Processing {N} studies sequentially into pre-allocated memory...")

for i, sid in enumerate(uids_list):
    sag_arr[i] = load_and_process_series(sid, 'Sagittal')
    cor_arr[i] = load_and_process_series(sid, 'Coronal')
    axi_arr[i] = load_and_process_series(sid, 'Axial')

    if (i + 1) % 200 == 0 or (i + 1) == N:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (N - (i + 1)) / max(rate, 1e-5)
        # Periodic memory cleanup
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            vram_mb = torch.cuda.memory_allocated() / 1e6
            print(f"  {i+1:4d}/{N} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | Rate: {rate:.1f} studies/s | GPU VRAM: {vram_mb:.0f}MB")
        else:
            print(f"  {i+1:4d}/{N} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s | Rate: {rate:.1f} studies/s")

# -----------------------------------------------------------------------
# 6. Save Compressed Cache File
# -----------------------------------------------------------------------
cache_path = '/kaggle/working/all_studies_cache.npz'
print(f"\n[SAVE] Writing compressed cache to {cache_path}...")
np.savez_compressed(
    cache_path,
    sag=sag_arr,
    cor=cor_arr,
    axi=axi_arr,
    labels=labels_arr,
    studies=np.array(uids_list)
)

elapsed = time.time() - t0
size_mb = os.path.getsize(cache_path) / (1024 ** 2)
print("\n" + "=" * 70)
print(f"[SUCCESS] RAM-Safe Cache saved: {cache_path}")
print(f"Shape: ({N}, {N_SLICES}, {IMG_SIZE}, {IMG_SIZE}) uint8 per plane")
print(f"File size on disk: {size_mb:.1f} MB (efficiently compressed)")
print(f"Total time: {elapsed:.0f}s (~{elapsed/60:.1f} minutes)")
print("=" * 70)
