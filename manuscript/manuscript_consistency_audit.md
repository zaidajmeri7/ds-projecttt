# Manuscript Consistency, Numerical Reconciliation, and Integrity Audit

**Audit Date:** 2026-10-31  
**Auditor:** Senior Medical ML Scientist, Epidemiological Methodologist, and Scientific Manuscript Editor  
**Document Target:** Phase 5 Publication Manuscript and Deliverables  

---

## 1. Audit Scope and Objectives
This independent consistency audit verifies that every numerical quantity, statistical test, citation record, methodological pipeline description, and scientific conclusion in the publication manuscript draft strictly matches the authoritative underlying data files produced across Phases 2, 3, 4, and 4.1.

---

## 2. Numerical Reconciliation and Verification Log

| Variable / Parameter | Source File | Authoritative Verified Value | Status |
| :--- | :--- | :--- | :--- |
| **Model SHA-256 Hash** | `phase4/model_freeze_hashes.csv` | `34bf9664529faa8cda5665907d7519f8fb0f3b82fcbd083dd33348c78403db80` | **VERIFIED 100% IDENTICAL** |
| **2020 Cleaned Total Records** | `results/tables/leakage_comparison.csv` | $N = 319,795$ ($255,836$ train + $63,959$ test) | **VERIFIED** |
| **2020 Test Positive Cases** | `results/tables/imbalance_comparison.csv` | $5,475$ positives ($8.56017\%$ prevalence) | **VERIFIED** |
| **2020 Test Unresampled XGBoost ROC-AUC** | `results/tables/imbalance_comparison.csv` | $0.837132$ (95% CI: $[0.831269, 0.842382]$) | **VERIFIED** |
| **2020 Test Unresampled XGBoost PR-AUC** | `results/tables/imbalance_comparison.csv` | $0.345524$ (95% CI: $[0.331256, 0.359020]$) | **VERIFIED** |
| **2020 Test Default ($\tau=0.50$) Recall** | `results/tables/imbalance_comparison.csv` | $9.2237\%$ ($505 / 5,475$ true positives) | **VERIFIED** |
| **2020 Test Default ($\tau=0.50$) Specificity** | `results/tables/imbalance_comparison.csv` | $99.3434\%$ ($58,100 / 58,484$ true negatives) | **VERIFIED** |
| **Coarse Threshold ($\tau_1$)** | `results/tables/threshold_comparison.csv` | $\tau = 0.079293$ (100-pt grid; recall $80.37\%$, spec $72.47\%$) | **VERIFIED** |
| **Fine Frozen Threshold ($\tau^*$)** | `phase4/temporal_validation_metrics.csv` | $\tau = 0.081612$ (1000-pt grid; recall $79.78\%$, spec $73.09\%$) | **VERIFIED** |
| **Phase 2 Typo "75.39% Specificity"** | `results/tables/demographic_ablation.csv` | Confirmed conflation with $\Delta \text{AUC}_{\text{demographics}} = 0.07539$ | **PERMANENTLY RETRACTED** |
| **Leaked Pre-Split SMOTE XGBoost AUC** | `results/tables/leakage_comparison.csv` | $0.966377$ (95% CI: $[0.965434, 0.967248]$) | **VERIFIED** |
| **Clean Training-Only SMOTE 1:1 AUC** | `results/tables/leakage_comparison.csv` | $0.809773$ (95% CI: $[0.803379, 0.815436]$) | **VERIFIED** |
| **Artifactual Leakage Inflation ($\Delta \text{AUC}$)** | `results/tables/leakage_comparison.csv` | $+0.156604$ ($0.966377 - 0.809773$, $p < 0.0001$) | **VERIFIED** |
| **Clean Brier vs 1:1 SMOTE Brier** | `phase3/statistical_comparisons.csv` | $0.065666$ vs $0.090699$ ($+38.12\%$ increase; $z = -51.82$, $p < 0.0001$) | **VERIFIED** |
| **Phase 4.1 "95% Brier Degradation"** | Audit trace | Conflated fold variance with test point estimate; corrected to $+38.1\%$ | **CORRECTED** |
| **2021 External Sample Size** | `phase4/temporal_validation_metrics.csv` | $N = 348,568$ complete cases (retention: $79.46\%$) | **VERIFIED** |
| **2021 Positives & Prevalence** | `phase4/temporal_validation_metrics.csv` | $28,444$ positives ($8.1602\%$ unweighted; $6.1841\%$ survey-weighted) | **VERIFIED** |
| **2021 Primary Median Imputation AUC** | `phase4/temporal_validation_metrics.csv` | $0.840739$ (95% CI: $[0.834900, 0.846736]$) | **VERIFIED** |
| **2021 Primary Sensitivity & Specificity** | `phase4/temporal_validation_metrics.csv` | Recall: $80.68\%$; Specificity: $72.24\%$ | **VERIFIED** |
| **2021 Survey-Weighted Metrics** | `phase4_1/design_based_comparison.csv` | Sensitivity: $75.19\%$; Specificity: $80.17\%$; ROC-AUC: $0.8564$ | **VERIFIED** |
| **2021 Sensitivity NaN Routing** | `phase4_1/sleep_handling_comparison.csv` | Sensitivity: $90.18\%$; Specificity: $58.74\%$; ECE: $0.0514$ | **VERIFIED** |
| **2022 External Sample Size** | `phase4/temporal_validation_metrics.csv` | $N = 339,264$ complete cases (retention: $76.22\%$) | **VERIFIED** |
| **2022 Positives & Prevalence** | `phase4/temporal_validation_metrics.csv` | $30,256$ positives ($8.9181\%$ unweighted; $6.7438\%$ survey-weighted) | **VERIFIED** |
| **2022 Narrative Typo "30,265"** | `phase4/temporal_validation_metrics.csv` | Reconciled: exact count is $30,256$ ($30,256 + 309,008 = 339,264$) | **CORRECTED** |
| **2022 Full Empirical Model AUC** | `phase4/temporal_validation_metrics.csv` | $0.839718$ (95% CI: $[0.833840, 0.846118]$) | **VERIFIED** |
| **2022 Sensitivity & Specificity** | `phase4/temporal_validation_metrics.csv` | Recall: $81.96\%$; Specificity: $70.87\%$ | **VERIFIED** |
| **2022 Survey-Weighted Metrics** | `phase4_1/design_based_comparison.csv` | Sensitivity: $78.72\%$; Specificity: $78.79\%$; ROC-AUC: $0.8607$ | **VERIFIED** |
| **Temporal Stability Test (2021 vs 2020)** | `phase4_1/temporal_difference_uncertainty.csv` | $\Delta \text{AUC} = +0.003607$, 95% CI: $[-0.0049, +0.0121]$, $p = 0.407$ | **VERIFIED** |
| **Temporal Stability Test (2022 vs 2020)** | `phase4_1/temporal_difference_uncertainty.csv` | $\Delta \text{AUC} = +0.002586$, 95% CI: $[-0.0061, +0.0113]$, $p = 0.560$ | **VERIFIED** |

---

## 3. Methodological and Conceptual Checks

1. **Retrospective Target Boundary**:
   - The manuscript exclusively describes the outcome as **self-reported lifetime physician-diagnosed coronary heart disease or myocardial infarction status (`_MICHD`)**.
   - No instances of "predicting future heart attack", "10-year risk", "cardiac arrest mortality", or "survival prediction" remain in the text.

2. **Causal Language Elimination**:
   - All feature relationships are strictly described as statistical predictive associations.
   - Prohibited causal terms ("causes", "leads to", "prevents", "reduces risk by modifying") have been replaced with descriptive epidemiological language.

3. **Demographic vs Lifestyle Grouping (`GenHealth`)**:
   - `GenHealth` is explicitly categorized under Group C (Self-Reported Health Status), NOT behavioral lifestyle.
   - The manuscript notes that single-item categorical lifestyle survey indicators add only $+0.011$ AUC over demographics and health status, explicitly qualifying this as survey measurement coarseness rather than biological irrelevance.

4. **Age Disparity Dual-Phenomenon Framing**:
   - The Phase 4 claim that age disparity was "purely an operational failure of thresholding" is fully retracted.
   - The text presents both:
     - Real within-group ranking discrimination degradation in elderly adults ($\text{AUC} \approx 0.735$ in ages 70+ vs $\text{AUC} \approx 0.795$ in ages <50, $p < 0.0001$).
     - Severe operating-threshold sensitivity disparities ($\sim 31\%$ in young vs $\sim 93\%$ in elderly) driven by the 11-fold prevalence gradient ($1.6\%$ vs $19.2\%$).
   - Age-specific thresholds are discussed as an operational tradeoff rather than an unvalidated clinical mandate.

5. **2021 Sleep Duration Transparency**:
   - The manuscript openly discloses that `SleepTime` was rotated off the 2021 nationwide core survey.
   - The primary analysis uses a predefined median imputation rule ($7.0$ hours) derived strictly from 2020 development training data.
   - The native NaN routing analysis is explicitly reported as a sensitivity benchmark, and 2022 full empirical measurement confirms genuine temporal stability.

6. **Complex Survey Weighting**:
   - Distinguishes respondent-level unweighted metrics from population-weighted estimands.
   - Explains that post-stratification raking down-weights older adults, lowering population sensitivity while elevating specificity.
   - Taylor series linearization accounts for strata (`_STSTR`) and primary sampling units (`_PSU`), providing design-aware confidence intervals.

---

## 4. Citation and Bibliography Verification
- All 22 cited works in `references.bib` are confirmed authentic, with verified DOIs, author lists, and publication outlets.
- CDC BRFSS documentation is cited directly from official CDC sources.
- No fabricated citations, author names, or page ranges exist.

---

## 5. Audit Conclusion
**Overall Status: PASSED WITHOUT OUTSTANDING DISCREPANCIES.**  
The evidence base, numbers, tables, figures, and narrative claims are 100% reconciled and prepared for formal manuscript submission and peer review.
