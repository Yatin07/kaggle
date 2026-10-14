# RSNA Knee Abnormality Detection — Long-Term Project Memory

## 1. Verified Filesystem Layout & Folder Scheme
- **Root Directory:** `/kaggle/input/competitions/rsna-knee-abnormality-detection`
- **Volumetric Path Format:** `train_series/{StudyInstanceUID}/{SeriesInstanceUID}/{slice_filename}.dcm`
- **DICOM Slice Specifications:**
  - Dimensions: `512 x 512`
  - Dtype: `uint16` (Intensity range: ~0 to 915+)
  - Modality: `MR`
  - Codec: Native pydicom standard decompression (verified working without errors).

## 2. Series Metadata & Anatomical Plane Coverage
- **Columns in `train_series.csv`:**
  `['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']`
- **Optimal Selection Rule:** Sort by `Fluid_Sensitive == 1` then `Fat_Suppression == 1`.
- **Plane Coverage (100% Perfect):**
  - Sagittal: Exactly 4,407 series (1 per study)
  - Coronal: Exactly 4,407 series (1 per study)
  - Axial: Exactly 4,407 series (1 per study)
  - Total Selected: 13,221 series.

## 3. Target Classes & Ground Truth Distribution
Out of **4,407 total studies**, exactly **58 studies have ground-truth human annotations**. The remaining 4,349 studies have free-text radiology reports.

| # | Finding Name | Annotated Count | Class Prior (Mean) |
|---|---|---|---|
| 1 | `ACL` (Anterior Cruciate Ligament) | 58 / 4,407 | **0.4138** |
| 2 | `MCL` (Medial Collateral Ligament) | 58 / 4,407 | **0.1552** |
| 3 | `Medial Meniscus` tear | 58 / 4,407 | **0.4483** |
| 4 | `Lateral Meniscus` tear | 58 / 4,407 | **0.3966** |
| 5 | `Medial OA` (Osteoarthritis) | 58 / 4,407 | **0.2586** |
| 6 | `Lateral OA` (Osteoarthritis) | 58 / 4,407 | **0.1897** |
| 7 | `PF OA` (Patellofemoral OA) | 58 / 4,407 | **0.3621** |
| 8 | `Effusion` (Joint Effusion) | 58 / 4,407 | **0.6034** |
| 9 | `Synovitis` | 58 / 4,407 | **0.4655** |
| 10 | `Baker's` (Baker's Cyst) | 58 / 4,407 | **0.2069** |
| 11 | `Contusion` (Bone Contusion) | 58 / 4,407 | **0.3276** |
| 12 | `Fracture` | 58 / 4,407 | **0.3103** |

## 4. Cache & Memory Footprint
- **Annotated Cache:** 58 studies × 3 planes × 3 slices (224×224) ≈ **~34 MB** total memory footprint.
- **Cache Format:** Saved as `/kaggle/working/knee_25d_cache.npz` for instantaneous loading.

## 5. Wave & Script Mapping Matrix
- **Wave 1:** [`wave1_baseline.py`](file:///e:/Kaggle/wave1_baseline.py) — Path resolution, prior computation, initial submission verification.
- **Wave 2:** [`wave2_series_and_dicom.py`](file:///e:/Kaggle/wave2_series_and_dicom.py) — Medically optimal series selection & DICOM loading check.
- **Wave 3:** [`wave3_dataset_cache.py`](file:///e:/Kaggle/wave3_dataset_cache.py) — 2.5D slice extraction & caching to `.npz`.
- **Wave 4:** [`wave4_train_model.py`](file:///e:/Kaggle/wave4_train_model.py) — PyTorch Multi-Plane ResNet training & Macro ROC-AUC validation.
- **Wave 5:** [`wave5_inference.py`](file:///e:/Kaggle/wave5_inference.py) — Zero-internet test set inference & final `submission.csv`.
