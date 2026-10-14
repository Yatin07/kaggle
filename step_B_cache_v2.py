"""
RSNA Knee — Improvement Module B: 2.5D Cache v2
WHY DIFFERENT FROM v1:
  - 7 equidistant slices per series (not 3 central) → captures full anatomy
  - CLAHE preprocessing per slice → enhances local MRI contrast
  - Uses ALL 4,407 studies (not just 58 labeled)
  - Saves separate cache: 'all_studies_cache.npz'
Runtime: ~30-45 minutes on Kaggle CPU. Run ONCE, reuse for many training runs.
"""
import os
import time
import numpy as np
import pandas as pd
import pydicom
import cv2

print("=" * 70)
print("RSNA Knee: Cache v2 — Full Study Set (7 slices, CLAHE)")
print("=" * 70)

comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

train_series_df = pd.read_csv(os.path.join(comp_dir, 'train_series.csv'))
pseudo_labels_path = '/kaggle/working/pseudo_labels.csv'

if not os.path.exists(pseudo_labels_path):
    raise FileNotFoundError("Run step_A_pseudo_labels.py first to generate pseudo_labels.csv!")

study_df = pd.read_csv(pseudo_labels_path)
print(f"Studies to cache: {len(study_df)}")

# Series Selection
sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
ascending = [True, True]
if 'Fluid_Sensitive' in train_series_df.columns:
    sort_cols.append('Fluid_Sensitive'); ascending.append(False)
if 'Fat_Suppression' in train_series_df.columns:
    sort_cols.append('Fat_Suppression'); ascending.append(False)

selected = train_series_df.sort_values(sort_cols, ascending=ascending).drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])
series_map = {}
for _, row in selected.iterrows():
    s = row['StudyInstanceUID']
    p = row['Anatomical_Plane']
    if s not in series_map:
        series_map[s] = {}
    series_map[s][p] = row['SeriesInstanceUID']

# CLAHE contrast enhancer for MRI
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def load_volume_v2(study_id, plane, n_slices=7, img_size=224):
    """Load n_slices equidistant slices from a series with CLAHE preprocessing."""
    if study_id not in series_map or plane not in series_map[study_id]:
        return np.zeros((n_slices, img_size, img_size), dtype=np.float32)
    
    series_id = series_map[study_id][plane]
    series_path = os.path.join(comp_dir, 'train_series', str(study_id), str(series_id))
    if not os.path.exists(series_path):
        return np.zeros((n_slices, img_size, img_size), dtype=np.float32)
    
    files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')])
    if len(files) == 0:
        return np.zeros((n_slices, img_size, img_size), dtype=np.float32)
    
    # Pick n_slices equidistant indices across the full volume
    indices = np.linspace(0, len(files) - 1, n_slices, dtype=int)
    
    channels = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(files[idx])
            img = dcm.pixel_array.astype(np.float32)
            
            # Normalize to uint8 for CLAHE
            mn, mx = img.min(), img.max()
            img_u8 = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
            
            # Apply CLAHE for local contrast enhancement
            img_clahe = clahe.apply(img_u8).astype(np.float32) / 255.0
            
            img_resized = cv2.resize(img_clahe, (img_size, img_size), interpolation=cv2.INTER_AREA)
        except Exception:
            img_resized = np.zeros((img_size, img_size), dtype=np.float32)
        channels.append(img_resized)
    
    return np.stack(channels, axis=0)  # (n_slices, 224, 224)


N_SLICES = 7
IMG_SIZE = 224
TARGET_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
               'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
               'Synovitis', "Baker's", 'Contusion', 'Fracture']

all_sag, all_cor, all_axi, all_labels, all_uids = [], [], [], [], []

t0 = time.time()
print(f"\n[CACHE] Processing {len(study_df)} studies ({N_SLICES} slices/plane, CLAHE)...")

for i, (_, row) in enumerate(study_df.iterrows()):
    study_id = row['StudyInstanceUID']
    all_sag.append(load_volume_v2(study_id, 'Sagittal', N_SLICES, IMG_SIZE))
    all_cor.append(load_volume_v2(study_id, 'Coronal', N_SLICES, IMG_SIZE))
    all_axi.append(load_volume_v2(study_id, 'Axial', N_SLICES, IMG_SIZE))
    all_labels.append([row.get(c, -1) for c in TARGET_COLS])
    all_uids.append(study_id)
    if (i + 1) % 500 == 0 or (i + 1) == len(study_df):
        elapsed = time.time() - t0
        print(f"  {i+1:4d}/{len(study_df)} processed | elapsed: {elapsed:.0f}s")

cache_path = '/kaggle/working/all_studies_cache.npz'
np.savez_compressed(
    cache_path,
    sag=np.array(all_sag, dtype=np.float32),
    cor=np.array(all_cor, dtype=np.float32),
    axi=np.array(all_axi, dtype=np.float32),
    labels=np.array(all_labels, dtype=np.float32),
    studies=np.array(all_uids)
)

elapsed = time.time() - t0
size_mb = os.path.getsize(cache_path) / (1024 ** 2)
print("\n" + "=" * 70)
print(f"[SUCCESS] Full cache saved: {cache_path}")
print(f"Shapes: ({len(study_df)}, {N_SLICES}, {IMG_SIZE}, {IMG_SIZE}) | {size_mb:.1f} MB | {elapsed:.0f}s")
print("=" * 70)
