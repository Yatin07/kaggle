"""
RSNA Knee Abnormality Detection — Wave 2: Series Selection & DICOM Decompression Test
Run this cell in your Kaggle Notebook.
"""
import os
import glob
import pandas as pd
import numpy as np
import pydicom

print("=" * 70)
print("RSNA Knee: Wave 2 — Series Selection & DICOM Slice Loader")
print("=" * 70)

# 1. Locate competition root
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

train_series_path = os.path.join(comp_dir, 'train_series.csv')
train_series_df = pd.read_csv(train_series_path)
print(f"Loaded train_series.csv: {train_series_df.shape[0]} series rows")
print("Columns in train_series.csv:", list(train_series_df.columns))

# 2. Series Selection Algorithm (1 Sagittal, 1 Coronal, 1 Axial per study)
# We group by StudyInstanceUID and Anatomical_Plane, selecting the best candidate series per plane
def select_best_series(df):
    # If SeriesDescription exists, rank sequences (favoring FS, T2, PD, STIR)
    def rank_series(row):
        score = 0
        desc = str(row.get('SeriesDescription', '')).lower()
        if 'fs' in desc or 'fat' in desc or 'supp' in desc:
            score += 10
        if 't2' in desc:
            score += 5
        if 'pd' in desc:
            score += 4
        return score

    if 'SeriesDescription' in df.columns:
        df['rank_score'] = df.apply(rank_series, axis=1)
        selected = df.sort_values(['StudyInstanceUID', 'Anatomical_Plane', 'rank_score'], ascending=[True, True, False])
    else:
        selected = df

    # Drop duplicate planes per study to keep exactly 1 series per plane
    selected = selected.drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])
    return selected

selected_df = select_best_series(train_series_df)
print(f"\n[SELECTION] Filtered to 1 series per plane: {len(selected_df)} series selected")
print("Plane counts in selected set:")
print(selected_df['Anatomical_Plane'].value_counts().to_string())

# 3. Test DICOM Slice Reading on a Real Study
sample_study_id = selected_df['StudyInstanceUID'].iloc[0]
sample_series_id = selected_df['SeriesInstanceUID'].iloc[0]
sample_plane = selected_df['Anatomical_Plane'].iloc[0]

study_dir = os.path.join(comp_dir, 'train_series', str(sample_study_id), str(sample_series_id))
if not os.path.exists(study_dir):
    # In case train_series is flat: train_series/{series_id}
    study_dir = os.path.join(comp_dir, 'train_series', str(sample_series_id))

print(f"\n[DICOM TEST] Inspecting Series Folder: {study_dir}")
if os.path.exists(study_dir):
    dcm_files = sorted([os.path.join(study_dir, f) for f in os.listdir(study_dir) if f.endswith('.dcm')])
    print(f"Total slices found in series: {len(dcm_files)}")
    
    if len(dcm_files) > 0:
        # Load middle slice
        mid_idx = len(dcm_files) // 2
        mid_dcm = pydicom.dcmread(dcm_files[mid_idx])
        pixel_array = mid_dcm.pixel_array
        
        print(f"Middle slice ({mid_idx}) loaded successfully!")
        print(f"  Shape: {pixel_array.shape}")
        print(f"  Dtype: {pixel_array.dtype}")
        print(f"  Value Range: min={pixel_array.min()}, max={pixel_array.max()}")
        print(f"  Modality: {getattr(mid_dcm, 'Modality', 'N/A')}")
        print("  Decompression: OK (No missing codec error!)")
else:
    print(f"Direct folder path check: searching train_series for {sample_study_id}...")
    for root, dirs, files in os.walk(os.path.join(comp_dir, 'train_series')):
        if any(f.endswith('.dcm') for f in files):
            print(f"Found DICOMs at: {root} ({len(files)} files)")
            sample_file = os.path.join(root, files[0])
            dcm = pydicom.dcmread(sample_file)
            print(f"Test slice shape: {dcm.pixel_array.shape}, dtype: {dcm.pixel_array.dtype}")
            break

print("=" * 70)
print("Wave 2 verification complete! Ready for 2.5D slice caching.")
