"""
RSNA Knee Abnormality Detection — Zero-to-Hero Starter & Exploration Baseline
Wave 1: Verification, Exploration & Pipeline Validation
"""
import os
import glob
import pandas as pd
import numpy as np

print("=" * 70)
print("RSNA Knee Abnormality Detection: Wave 1 Execution & Exploration")
print("=" * 70)

# 1. Foolproof Path Resolution: find folder containing sample_submission.csv
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

if comp_dir is None:
    raise FileNotFoundError("Could not find sample_submission.csv inside /kaggle/input")

print(f"\n[FOUND] Competition Root: {comp_dir}")
files_present = os.listdir(comp_dir)
print("Files & directories in root:", files_present)

# 2. Inspect Sample Submission & Target Columns
sample_sub_path = os.path.join(comp_dir, 'sample_submission.csv')
sample_sub = pd.read_csv(sample_sub_path)
target_cols = [c for c in sample_sub.columns if c != 'StudyInstanceUID']

print(f"\n[SUBMISSION] Target findings ({len(target_cols)} classes):")
for idx, col in enumerate(target_cols, 1):
    print(f"  {idx:2d}. {col}")

# 3. Read Test CSV
test_path = os.path.join(comp_dir, 'test.csv')
test_df = pd.read_csv(test_path)
print(f"\n[TEST] Sample test studies loaded: {test_df.shape[0]} rows (Full ~1,300 evaluated on commit)")

# 4. Explore Training Labels & Class Prevalence
train_path = os.path.join(comp_dir, 'train.csv')
sub = pd.DataFrame({'StudyInstanceUID': test_df['StudyInstanceUID']})

if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    print(f"\n[TRAIN] train.csv loaded: {train_df.shape[0]} total studies, {train_df.shape[1]} columns")
    
    print("\n[LABEL PREVALENCE] Available positive findings & label coverage:")
    for col in target_cols:
        if col in train_df.columns:
            non_null = train_df[col].notna().sum()
            if non_null > 0:
                prior = float(train_df[col].dropna().mean())
            else:
                prior = 0.5
            sub[col] = prior
            print(f"  {col:<42}: prior={prior:.4f} ({non_null}/{len(train_df)} annotated)")
        else:
            sub[col] = 0.5
else:
    for col in target_cols:
        sub[col] = 0.5

# 5. Explore Series Metadata (Anatomical Planes & Modalities)
train_series_path = os.path.join(comp_dir, 'train_series.csv')
if os.path.exists(train_series_path):
    ts_df = pd.read_csv(train_series_path)
    print(f"\n[SERIES METADATA] train_series.csv loaded: {len(ts_df)} series across {ts_df['StudyInstanceUID'].nunique()} studies")
    
    # Show breakdown by plane and description
    for candidate_col in ['Anatomical_Plane', 'SeriesDescription', 'SeriesType']:
        if candidate_col in ts_df.columns:
            print(f"\nTop values for '{candidate_col}':")
            print(ts_df[candidate_col].value_counts().head(8).to_string())

# 6. Generate submission.csv
output_file = '/kaggle/working/submission.csv'
sub.to_csv(output_file, index=False)
print("\n" + "=" * 70)
print(f"[SUCCESS] Generated: {output_file} (shape: {sub.shape})")
print(sub.head(3))
print("=" * 70)
print("Pipeline verified! Commit notebook now to record your baseline score on Kaggle.")
