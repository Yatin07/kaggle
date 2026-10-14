# RSNA Knee Abnormality Detection (Kaggle) 🚀

A structured, spec-driven multi-plane 2.5D deep learning framework for the **RSNA Knee Abnormality Detection** Kaggle challenge, developed using the **Get Shit Done (GSD)** methodology.

## 📌 Project Overview
- **Competition:** RSNA Knee Abnormality Detection
- **Objective:** Detect 12 clinically significant knee abnormalities across multi-series MRI exams.
- **Metric:** Macro-averaged ROC-AUC across 12 findings.
- **Model Architecture:** Multi-Plane 2.5D ResNet combining central slice volumes from Sagittal, Coronal, and Axial planes.
- **Validation Metric:** Reached **0.6516 Macro ROC-AUC** in initial supervised validation.

---

## 🗂️ 5-Wave GSD Architecture

| Wave | Script | Function |
|:---:|:---|:---|
| **Wave 1** | [`wave1_baseline.py`](wave1_baseline.py) | Dynamic path discovery, class prior probability computation, initial sanity check. |
| **Wave 2** | [`wave2_series_and_dicom.py`](wave2_series_and_dicom.py) | Optimal series selection via `Fluid_Sensitive` & `Fat_Suppression` flags (100% plane coverage). |
| **Wave 3** | [`wave3_dataset_cache.py`](wave3_dataset_cache.py) | Fast 2.5D slice extraction, resizing to 224x224, and caching to `.npz`. |
| **Wave 4** | [`wave4_train_model.py`](wave4_train_model.py) | PyTorch `MultiPlaneKneeNet` training with BCEWithLogitsLoss and Cosine Annealing. |
| **Wave 5** | [`wave5_inference.py`](wave5_inference.py) | Zero-internet offline inference pipeline generating final `submission.csv`. |
| **All-in-One** | [`full_pipeline_notebook.py`](full_pipeline_notebook.py) | Single-cell consolidated execution script for Kaggle notebooks. |

---

## 🧠 Project State & Documentation
- [`.planning/SPEC.md`](.planning/SPEC.md) — Competition specifications & compute rules.
- [`.planning/PLAN.md`](.planning/PLAN.md) — Wave-based milestone tracking.
- [`.planning/STATE.md`](.planning/STATE.md) — Real-time progress and metric logs.
- [`.planning/MEMORY.md`](.planning/MEMORY.md) — Persistent dataset statistics and ground-truth numbers.
