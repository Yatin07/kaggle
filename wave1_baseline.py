"""
RSNA Knee Abnormality Detection — Wave 1: Baseline Pipeline & Verification
Run in Kaggle Notebook (GPU T4 or CPU, Internet: OFF).
"""
import os
import glob
import pandas as pd
import numpy as np

print("=" * 70)
print("RSNA Knee: Wave 1 — Baseline Pipeline & Verification")
print("=" * 70)

# 1. Path Resolution
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

if comp_dir is None:
    raise FileNotFoundError("Could not find sample_submission.csv inside /kaggle/input")

print(f"\n[FOUND] Competition Root: {comp_dir}")

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

# 5. Generate submission.csv
output_file = '/kaggle/working/submission.csv'
sub.to_csv(output_file, index=False)
print("\n" + "=" * 70)
print(f"[SUCCESS] Wave 1 Output Generated: {output_file} (shape: {sub.shape})")
print(sub.head(3))
print("=" * 70)
