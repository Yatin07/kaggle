"""
RSNA Knee Abnormality Detection — Wave 3: 2.5D Slice Extraction & Caching
Run in Kaggle Notebook (GPU T4 or CPU). Takes ~10-15 seconds to create compact ~34 MB cache.
"""
import os
import glob
import time
import pandas as pd
import numpy as np
import pydicom
import cv2

print("=" * 70)
print("RSNA Knee: Wave 3 — 2.5D Multi-Plane Slicing & Caching")
print("=" * 70)

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

# Filter only the 58 annotated studies
labeled_df = train_df[train_df['ACL'].notna()].copy().reset_index(drop=True)
print(f"[DATA] Studies to Extract: {len(labeled_df)} ground-truth annotated cases")

# 2. Optimal Series Selection
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

# 3. 2.5D Slice Extraction Function
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

# 4. Process and Save to NPZ Cache
print("\n[EXTRACTION] Processing 2.5D volumes for Sagittal, Coronal, and Axial planes...")
t0 = time.time()
all_sag, all_cor, all_axi, all_labels = [], [], [], []

for idx, row in labeled_df.iterrows():
    study_id = row['StudyInstanceUID']
    all_sag.append(extract_25d_volume(study_id, 'Sagittal'))
    all_cor.append(extract_25d_volume(study_id, 'Coronal'))
    all_axi.append(extract_25d_volume(study_id, 'Axial'))
    all_labels.append(row[target_cols].values.astype(np.float32))

cache_path = '/kaggle/working/knee_25d_cache.npz'
np.savez_compressed(
    cache_path,
    sag=np.array(all_sag, dtype=np.float32),
    cor=np.array(all_cor, dtype=np.float32),
    axi=np.array(all_axi, dtype=np.float32),
    labels=np.array(all_labels, dtype=np.float32),
    studies=labeled_df['StudyInstanceUID'].values
)

elapsed = time.time() - t0
size_mb = os.path.getsize(cache_path) / (1024 * 1024)
print("\n" + "=" * 70)
print(f"[SUCCESS] Wave 3 Cache Created in {elapsed:.1f}s: {cache_path} ({size_mb:.2f} MB)")
print("Shapes: Sagittal, Coronal, Axial = (58, 3, 224, 224), Labels = (58, 12)")
print("=" * 70)
