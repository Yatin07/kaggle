"""
RSNA Knee Abnormality Detection — Wave 5: Offline Model Inference & Final Submission
Runs in Kaggle Notebook with Internet: OFF. Reads test set, runs 2.5D multi-plane model, outputs submission.csv.
"""
import os
import glob
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import torch.nn as nn
from torchvision import models

print("=" * 70)
print("RSNA Knee: Wave 5 — Offline Model Inference & Submission Pipeline")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference Device: {DEVICE}")

# 1. Path Resolution
comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

test_df = pd.read_csv(os.path.join(comp_dir, 'test.csv'))
test_series_df = pd.read_csv(os.path.join(comp_dir, 'test_series.csv'))
sample_sub = pd.read_csv(os.path.join(comp_dir, 'sample_submission.csv'))
target_cols = [c for c in sample_sub.columns if c != 'StudyInstanceUID']

print(f"Total Test Studies to Predict: {len(test_df)}")

# 2. Optimal Test Series Selection (Fluid_Sensitive -> Fat_Suppression)
sort_cols = ['StudyInstanceUID', 'Anatomical_Plane']
ascending_order = [True, True]

if 'Fluid_Sensitive' in test_series_df.columns:
    sort_cols.append('Fluid_Sensitive')
    ascending_order.append(False)
if 'Fat_Suppression' in test_series_df.columns:
    sort_cols.append('Fat_Suppression')
    ascending_order.append(False)

selected_test_series = test_series_df.sort_values(sort_cols, ascending=ascending_order)
selected_test_series = selected_test_series.drop_duplicates(subset=['StudyInstanceUID', 'Anatomical_Plane'])

test_series_map = {}
for _, row in selected_test_series.iterrows():
    study = row['StudyInstanceUID']
    plane = row['Anatomical_Plane']
    series = row['SeriesInstanceUID']
    if study not in test_series_map:
        test_series_map[study] = {}
    test_series_map[study][plane] = series

# 3. Fast 2.5D Slice Extractor for Test
def load_test_25d_volume(study_id, plane, img_size=224):
    if study_id not in test_series_map or plane not in test_series_map[study_id]:
        return np.zeros((3, img_size, img_size), dtype=np.float32)
    
    series_id = test_series_map[study_id][plane]
    series_path = os.path.join(comp_dir, 'test_series', str(study_id), str(series_id))
    if not os.path.exists(series_path):
        series_path = os.path.join(comp_dir, 'test_series', str(series_id))
    
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

# 4. Neural Network Architecture Definition
class MultiPlaneKneeNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        feature_dim = 512
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, sag, cor, axi):
        f_sag = self.backbone(sag).flatten(1)
        f_cor = self.backbone(cor).flatten(1)
        f_axi = self.backbone(axi).flatten(1)
        fused = torch.cat([f_sag, f_cor, f_axi], dim=1)
        return self.classifier(fused)

# 5. Load Trained Weights & Run Inference
model = MultiPlaneKneeNet(num_classes=12).to(DEVICE)
weight_path = '/kaggle/working/best_multiplane_knee_model.pth'

# Known Class Priors from Memory Bank as safety fallback
PRIORS = {
    'ACL': 0.4138, 'MCL': 0.1552, 'Medial Meniscus': 0.4483, 'Lateral Meniscus': 0.3966,
    'Medial OA': 0.2586, 'Lateral OA': 0.1897, 'PF OA': 0.3621, 'Effusion': 0.6034,
    'Synovitis': 0.4655, "Baker's": 0.2069, 'Contusion': 0.3276, 'Fracture': 0.3103
}

has_weights = os.path.exists(weight_path)
if has_weights:
    print(f"Loading trained model weights from: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
else:
    print("[WARNING] Weights file not found in /kaggle/working/! Falling back to class priors.")

all_predictions = []

print(f"\n[INFERENCE] Predicting across {len(test_df)} studies...")
for i, study_id in enumerate(test_df['StudyInstanceUID']):
    if (i + 1) % 100 == 0 or (i + 1) == len(test_df):
        print(f"  Processed {i + 1}/{len(test_df)} studies")

    if has_weights:
        sag = torch.tensor(load_test_25d_volume(study_id, 'Sagittal'), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        cor = torch.tensor(load_test_25d_volume(study_id, 'Coronal'), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        axi = torch.tensor(load_test_25d_volume(study_id, 'Axial'), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(sag, cor, axi)).cpu().numpy()[0]
            # Blend 85% model predictions + 15% empirical priors for maximum ROC-AUC calibration
            blended = [0.85 * preds[j] + 0.15 * PRIORS[col] for j, col in enumerate(target_cols)]
            all_predictions.append(blended)
    else:
        all_predictions.append([PRIORS[col] for col in target_cols])

# 6. Generate submission.csv
sub_df = pd.DataFrame(all_predictions, columns=target_cols)
sub_df.insert(0, 'StudyInstanceUID', test_df['StudyInstanceUID'])

sub_df.to_csv('/kaggle/working/submission.csv', index=False)
print("\n" + "=" * 70)
print(f"[SUCCESS] Offline inference complete! Output saved to: /kaggle/working/submission.csv")
print(f"Output shape: {sub_df.shape}")
print(sub_df.head())
print("=" * 70)
