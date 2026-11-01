# Manuscript Build and Reproducibility Guide

## 1. Overview
This repository contains the complete publication-oriented research manuscript, verified tables, figures, statistical audit files, and LaTeX source code for:

**"Beyond Random Splits: Leakage, Imbalance Handling, Calibration, and Temporal Generalization in BRFSS-Based Heart Disease Classification"**

---

## 2. Directory Structure
```
├── manuscript/ (or phase5/manuscript/)
│   ├── main.tex                       # Complete LaTeX manuscript source
│   ├── references.bib                 # Verified BibTeX bibliography (22 primary sources)
│   ├── claim_evidence_matrix.csv      # Traceability matrix mapping claims to evidence
│   ├── reference_verification.csv     # DOI and venue verification records
│   ├── reporting_guideline_checklist.csv # STROBE and TRIPOD+AI compliance checklist
│   ├── manuscript_consistency_audit.md   # Final numerical and conceptual consistency audit
│   ├── supplementary_appendix.md      # Expanded methods, crosswalks, and mathematical derivations
│   ├── tables/                        # Publication tables in CSV and LaTeX
│   │   ├── table1_cohort_summary.csv
│   │   ├── table2_feature_groups.csv
│   │   ├── table3_imbalance_methods.csv
│   │   ├── table4_leakage_audit.csv
│   │   ├── table5a_thresholds.csv
│   │   ├── table5b_calibration.csv
│   │   ├── table6a_feature_ablation.csv
│   │   ├── table6b_permutation_importance.csv
│   │   ├── table7_age_disparities.csv
│   │   ├── table7b_subgroup_transfer.csv
│   │   ├── table8_temporal_validation.csv
│   │   ├── table9_survey_weighting.csv
│   │   ├── table9b_design_uncertainty.csv
│   │   ├── table10_complete_case_audit.csv
│   │   └── table10b_sleep_sensitivity.csv
│   └── figures/                       # Publication figures (300 DPI PNG)
│       ├── study_design_workflow.png            # Figure 1: Study Design & Flowchart
│       ├── leakage_comparison.png               # Figure 2: Clean vs Leaked Resampling
│       ├── imbalance_metric_comparison.png      # Figure 3: Imbalance Methods Comparison
│       ├── roc_curves_by_wave.png               # Figure 4a: ROC Curves Across Waves
│       ├── pr_curves_by_wave.png                # Figure 4b: PR Curves Across Waves
│       ├── threshold_metric_curves.png          # Figure 5: Operating Threshold Dynamics
│       ├── calibration_curves_by_wave.png       # Figure 6: Reliability Calibration Curves
│       ├── subgroup_temporal_generalization.png # Figure 7: Age & Demographic Disparities
│       ├── temporal_confusion_matrices.png      # Figure 8: Multi-Wave Confusion Matrices
│       ├── age_transfer_heatmap.png             # Supplementary Age Transfer Matrix
│       ├── feature_distribution_shift.png       # Supplementary Covariate Shift
│       └── shap_summary.png                     # Supplementary SHAP Beeswarm Plot
```

---

## 3. LaTeX Compilation Instructions

### Prerequisites
- TeX Live 2023+ or MacTeX or MikTeX
- Required packages: `amsmath`, `amssymb`, `graphicx`, `booktabs`, `multirow`, `hyperref`, `cite`, `geometry`, `caption`, `subcaption`

### Local Command-Line Build
```bash
# Step 1: Initial compile
pdflatex main.tex

# Step 2: Bibliography processing
bibtex main

# Step 3: Two cross-reference resolution passes
pdflatex main.tex
pdflatex main.tex
```

### Overleaf Compilation
1. Zip the `manuscript/` directory (including `main.tex`, `references.bib`, and the `figures/` subfolder).
2. Create a new blank project on Overleaf and upload the zip archive.
3. Select `pdfLaTeX` as the compiler in Project Settings.
4. Click `Recompile`.

---

## 4. Computational Reproducibility
The data processing, modeling, and evaluation pipelines can be reproduced using Python 3.10+:

```bash
# Install core dependencies
pip install numpy pandas scikit-learn xgboost imbalanced-learn scipy matplotlib seaborn shap

# Execute audit and reproduction scripts
python phase3/audit_dataset_integrity.py
python phase3/reconcile_headline_metrics.py
python phase4/run_temporal_evaluation.py
python phase4_1/generate_phase4_1_audits.py
```

All trained binary models are hash-verified (`xgb_model.joblib` SHA-256: `34bf9664529faa8cda5665907d7519f8fb0f3b82fcbd083dd33348c78403db80`).
