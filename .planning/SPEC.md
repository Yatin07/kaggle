# RSNA Knee Abnormality Detection — Project Specification

## 1. Objective & Overview
- **Competition:** RSNA Knee Abnormality Detection (Kaggle)
- **Goal:** Predict the presence of 12 clinically significant knee abnormalities from multi-series MRI examinations.
- **Metric:** Macro-averaged ROC-AUC across 12 binary findings.
- **Timeline:**
  - Oct 15, 2026: Team Merger & Entry Deadline (**CRITICAL: Must join now**).
  - Oct 22, 2026 (11:59 PM UTC): Final Submission Deadline.

## 2. Hard Constraints & Environment Rules
1. **Local Storage:** The dataset is ~570 GB. **DO NOT** download image data to local machine. All image extraction, caching, and model training run on Kaggle GPU instances (GPU T4 or T4 x2).
2. **Inference Requirements:**
   - Internet Access: **OFF** (Must not require external API calls or on-the-fly package downloads).
   - Maximum Runtime: **9 Hours**.
   - Input Structure: Code competition swaps hidden test data (~1,300 studies) at scoring time. Never hardcode test set lengths.
3. **Data Security (Rule 4.b):**
   - MRI scans and radiology reports must NOT be uploaded to external unauthorized cloud services.
   - Code and summaries only are kept locally; model inference and data processing run directly on Kaggle.

## 3. The 12 Abnormalities (Target Columns)
1. Anterior Cruciate Ligament (ACL) tear/injury
2. Posterior Cruciate Ligament (PCL) tear/injury
3. Medial Collateral Ligament (MCL) tear/injury
4. Lateral Collateral Ligament (LCL) tear/injury
5. Medial Meniscus tear
6. Lateral Meniscus tear
7. Medial Compartment Osteoarthritis
8. Lateral Compartment Osteoarthritis
9. Patellofemoral Compartment Osteoarthritis
10. Joint Effusion
11. Synovitis
12. Baker's Cyst / Bone Contusion / Fracture

## 4. Modeling Architecture: 2.5D Multi-Plane Pipeline
- Knee MRIs are 3D volumetric scans across 3 anatomical planes: Sagittal, Coronal, Axial.
- Standard 3D CNNs are too heavy for Kaggle's 9-hour limit.
- **2.5D Approach:** For each plane, select key fluid-sensitive/fat-suppressed series, take central slices, stack 3 adjacent slices as RGB channels, and feed into a lightweight 2D backbone (e.g., `timm` EfficientNet-B0 / ResNet34) with multi-label classification head.
