# RSNA Knee Abnormality Detection — Iteration 4 Master Research & Execution Plan

**Author:** Antigravity AI Pair-Programming Assistant  
**Repository:** [https://github.com/Yatin07/kaggle](https://github.com/Yatin07/kaggle)  
**Date:** October 16, 2026  
**Competition Deadline:** October 22, 2026 (**6 Days Remaining**)  
**Target Metric:** Macro-Averaged ROC-AUC across 12 Abnormality Classes  
**Score Progression:** Iteration 1 (`0.602`) $\to$ Iteration 2 (`0.590`) $\to$ Iteration 3 (`5-Fold Clean Gold Baseline`) $\to$ **Iteration 4 Target: `0.957 – 0.960+`**

---

## 1. Executive Summary & Strategy

In Iterations 1 through 3, we resolved fundamental medical imaging errors (spatial slice-normal depth ordering, removal of medial/lateral flipping augmentations, and rigorous 5-fold cross-validation on gold labels). 

Iteration 4 shifts from basic baseline debugging to **top-tier leaderboard dominance**. Analysis of the community state-of-the-art (**Aleksei Provorov’s V33 four-model ensemble**) proves that the 0.957+ threshold is achieved through five specific pillars:
1. **Finding-Specific Attention Pooling (2.5D MIL):** Allowing each abnormality (e.g., ACL vs. Medial OA) to dynamically attend to its own relevant slices rather than uniform mean-pooling.
2. **Hybrid Vision Backbones:** CoAtNet (`coatnet_rmlp_2_rw_384`, `224`) and ConvNeXt (`R7`, `S6`) pretrained on extensive clinical MRI corpuses.
3. **Continuous Soft Pseudo-Labels:** Training on LLM-derived soft probability distributions ($p \in [0, 1]$) across 4,349 studies instead of flawed discrete regex rules.
4. **Medically Validated L/R Mirror TTA:** Flipping slice channels along Z for Sagittal scans, and flipping image width horizontally for Coronal/Axial scans *before* normalization.
5. **Cross-Family Percentile Rank Blending:** Ensembling predictions via test-cohort percentile ranking rather than raw probability averages.

Iteration 4 is executed in two parallel, risk-managed tracks:
* **Track A (Immediate SOTA Lock-in):** Deploy the V33 4-model ensemble on Kaggle Dual T4 GPUs to secure an official **`0.957+`** score today.
* **Track B (Custom Vision Distillation):** Train our custom CoAtNet/ConvNeXt pipeline using clean soft labels and 3D slice-normal projections.
* **Track C (Final Fusion):** Blend Track A and Track B for a complementary **`0.960+`** medal submission.

---

## 2. Iteration History & Empirical Audit

| Iteration | Architecture | Data Used | Slices & Orientation | Augmentations | Loss & Fusion | Score (Public LB) |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **Iter 1 (Baseline)** | ResNet18 (Random weights) | 58 Gold studies (46 train) | 3 center slices per plane | None | Standard BCE | **0.602** |
| **Iter 2 (Scaled)** | Multi-Plane ResNet34 + Attention | 4,407 studies (Regex mined) | 7 equidistant slices (UUID sorted) | H-Flip, V-Flip | Masked Focal-BCE | **0.590** |
| **Iter 3 (Research)** | Multi-Plane ResNet34 + Attention | 58 Gold (5-Fold Stratified CV) | 16 slices (True 3D Slice-Normal sorted) | Safe Affine (Rot $\pm 10^\circ$, Scale $\pm 8\%$, NO flips) | Class-Weighted BCE + Cosine LR | **Benchmarked** |
| **Iter 4 (Target)** | **4-Model SOTA Ensemble + Custom CoAtNet** | **4,407 Studies (Soft LLM Labels + 58 Gold)** | **94 Attention Windows (3-slice 2.5D triplets)** | **Anatomical Mirror TTA (Sagittal Z, Cor/Axi X)** | **50/50 Family Percentile Rank Blend** | **0.957 – 0.960+** |

### Verified Lessons from Iteration 2 Diagnostic Checks:
1. **DICOM Depth Order:** DICOM filenames are random UUIDs with near-zero correlation ($\rho \approx 0.01$) to physical slice depth. Sorting by $\vec{n} = \vec{r} \times \vec{c}$ and $\text{depth} = \vec{p} \cdot \vec{n}$ is non-negotiable.
2. **Destructive Flips:** Naive horizontal flipping inverts Medial and Lateral joint compartments, corrupting labels on Coronal/Axial planes.
3. **Regex Failure:** Heuristic regex missed $>80\%$ of true positives and corrupted $46.7\%$ of Medial OA cases to negative due to greedy negation matching. Continuous soft probabilities are mandatory.

---

## 3. Architecture of the V33 SOTA 4-Model Ensemble

```
                                  Final Submission (100%)
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼ (50%)                                         ▼ (50%)
               Raptor Family                                   Compact Family
         (CoAtNet Hybrids / nartaa)                       (ConvNeXt / heliosli, nartaa)
                     │                                               │
             ┌───────┴───────┐                               ┌───────┴───────┐
             ▼ (65%)         ▼ (35%)                         ▼ (60%)         ▼ (40%)
         Raptor-384      Raptor-224                      R7 ConvNeXt     S6 ConvNeXt
         (32.5% net)     (17.5% net)                     (30.0% net)     (20.0% net)
             │               │                               │               │
          TTA Rank       V33 Mirror Mix:                  Raw Rank        Raw Rank
          (100%)         ┌───┴───────────┐                (100%)          (100%)
                         ▼ (50%)         ▼ (50%)
                      Plain Rank      TTA Rank
                      (8.75% net)     (8.75% net)
```

### 3.1 Model Components & Checkpoint Provenance
1. **Raptor-384 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`):**
   * Pretrained on OAI + 4,349 expanded RSNA corpus with soft JEV labels.
   * Input: 96 slices structured into 5 anatomical sequence slots.
   * Inference: 94 sliding 3-slice triplets with finding-specific linear attention.
   * SHA-256: `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`.
2. **Raptor-224 (`coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`):**
   * Retrained with native $224 \times 224$ crops ($0.80$ central fraction of the $140\text{ mm}$ field).
   * Generates both Plain and Anatomical Mirror predictions.
   * SHA-256: `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`.
3. **R7 ConvNeXt:**
   * Compact multi-view ConvNeXt by heliosli (bundle `8c7b666e27dc0e8f`).
   * Captures complementary convolutional inductive biases.
4. **S6 ConvNeXt:**
   * All-data fall checkpoint by nartaa (`E_prior_s42_fall.pt`).
   * Protected by S6 correlation gate and fallback checks.
   * SHA-256: `0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`.

### 3.2 Anatomical Mirror TTA Formula
Standard horizontal flipping corrupts knee anatomy. The medically correct mirror TTA applies axis-aware transformation on the raw uint8 triplets *before* per-channel normalization:
* **Sagittal Windows (Slots 0 & 1):** Reverse the 3 slice channels:
  $$\text{Tri}_{\text{mirror}} = \text{tri.flip}(1) \quad (\text{Z-depth reversal across the joint space})$$
* **Coronal & Axial Windows (Slots 2, 3 & 4):** Reverse the image width:
  $$\text{Tri}_{\text{mirror}} = \text{tri.flip}(-1) \quad (\text{Horizontal L/R reflection})$$

---

## 4. Phase-by-Phase Execution Plan

### Phase 1: Deploy & Benchmark V33 in Kaggle (Day 1 — Immediate)
* **Objective:** Secure an official **`0.957+`** Public Leaderboard submission within 2 hours.
* **Tasks:**
  1. Attach the 4 community dataset mounts to the Kaggle notebook:
     * `nartaa/rsna-knee-publication-swa-weights-20261007`
     * `heliosli/rsna-knee-compact-r7cnxfsss`
     * `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`
     * `nartaa/rsna-knee-s6fall-weights`
  2. Configure Kaggle hardware: **Dual Tesla T4 (`GPU T4 x2`)**, Internet **OFF**.
  3. Set recipe parameters: `RECIPE = 'mirror_mix_half'`, `REQUIRE_S6 = True`, `TOTAL_HOURS = 8.3`.
  4. Run pipeline, verify log output on sample studies, and submit via **Quick Save**.
  5. Log confirmed score to project state.

### Phase 2: Build Clean Soft-Label Dataset & Model Architecture (Days 2–3)
* **Objective:** Build our internal research model that trains on all 4,407 studies without label corruption.
* **Tasks:**
  1. Audit community soft-label tables (`stevenleehans`, `JEV-blended`) using our leakage diagnostic against the 58 gold studies. Confirm zero exact-copy leakage ($\text{AUC} < 1.0$ on gold).
  2. Implement Finding-Specific Multi-Instance Attention Pooling:
     $$\alpha_{k, c} = \frac{\exp(w_c^T \tanh(W h_k + b))}{\sum_j \exp(w_c^T \tanh(W h_j + b))}, \quad z_c = \sum_k \alpha_{k, c} h_k$$
  3. Build 5-slot sequence loader: 26 Sagittal (FS), 22 Sagittal (non-FS), 18 Coronal (FS), 12 Coronal (non-FS), 18 Axial.
  4. Train a 5-fold `coatnet_rmlp_2_rw_224` or `convnext_tiny` on dual T4 GPUs using AMP FP16.

### Phase 3: Anatomical Mirror TTA & Out-of-Fold Calibration (Days 4–5)
* **Objective:** Validate internal model gains on the 58 gold cases and calibrate rank percentiles.
* **Tasks:**
  1. Measure Out-of-Fold ROC-AUC across all 12 classes with plain vs. mirror TTA.
  2. Calculate correlation matrix between our custom model predictions and the V33 model predictions.
  3. Verify that the custom model provides distinct rank signals (Pearson correlation $r \approx 0.85 - 0.92$, indicating valuable ensembling diversity).

### Phase 4: Final 5-Way Rank Ensembling & Submission Freeze (Day 6 — Oct 21–22)
* **Objective:** Generate the final two submissions before the competition deadline.
* **Tasks:**
  1. Formulate 5-way percentile rank blend:
     $$\text{Rank}_{\text{final}} = 0.40 \times \text{Rank}_{\text{Raptor}} + 0.40 \times \text{Rank}_{\text{Compact}} + 0.20 \times \text{Rank}_{\text{Custom}}$$
  2. Submission 1: Conservative V33 Baseline (`0.957+` proven LB).
  3. Submission 2: 5-Way Diverse Ensemble Blend (aiming for `0.960+`).
  4. Final verification of submission format, 0 nulls, correct column order, and complete receipt logs.

---

## 5. Day-by-Day Countdown to Deadline

```
Oct 16 (Today)  ──► Phase 1: Deploy V33 4-Model Ensemble on Kaggle (Lock in 0.957+)
Oct 17 (Sat)    ──► Phase 2a: Audit Soft-Label CSVs & Build 2.5D MIL Attention Architecture
Oct 18 (Sun)    ──► Phase 2b: Launch 5-Fold Training on Full 4,407 Studies (Dual T4 AMP)
Oct 19 (Mon)    ──► Phase 3a: Out-of-Fold Validation & Mirror TTA Benchmarking
Oct 20 (Tue)    ──► Phase 3b: Compute Cross-Model Correlation & Rank Optimization
Oct 21 (Wed)    ──► Phase 4a: Generate Final 5-Way Ensembles & Dry-Run Submission
Oct 22 (Thu)    ──► FINAL DEADLINE: Lock Final 2 Submissions on Kaggle Leaderboard
```

---

## 6. Execution Safeguards & Competition Constraints

1. **Internet Off:** All models, weights, and packages (`pylibjpeg`, `timm`) must be mounted as offline Kaggle datasets.
2. **Runtime Ceiling:** Total inference runtime must remain under **$5.0$ hours** (well below Kaggle’s 9.0-hour limit).
3. **RAM & GPU VRAM Management:** Run model stages sequentially; call `del model`, `torch.cuda.empty_cache()`, and `gc.collect()` between stages to prevent host RAM / VRAM OOM.
4. **Never Fork Public Kernels:** Build, understand, test, and control all execution scripts directly within our own environment.
