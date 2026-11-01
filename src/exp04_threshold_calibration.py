"""
Experiment 4: Threshold Selection & Probability Calibration
Project: Heart Disease Prediction and Risk-Factor Analysis
"""

import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from src.data_loader import load_clean_data, build_preprocessor, load_config
from src.metrics import compute_binary_metrics, compute_ece, compute_bootstrap_ci

def run_threshold_and_calibration():
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 4: THRESHOLD SELECTION AND PROBABILITY CALIBRATION", flush=True)
    print("=" * 70, flush=True)
    
    df, train_df, test_df, config = load_clean_data()
    seed = config["project"]["random_seed"]
    all_features = config["feature_groups"]["group_d_all_features"]
    num_cols = [c for c in config["features"]["numerical"] if c in all_features]
    cat_cols = [c for c in config["features"]["categorical"] if c in all_features]
    
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Generate Out-of-Fold (OOF) Probabilities using 5-Fold Stratified CV
    # -------------------------------------------------------------------------
    print("Step 1: Generating Out-Of-Fold (OOF) probabilities on training data...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(train_df))
    
    # We evaluate with our primary full model: XGBoost (and compare with LogisticRegression)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        f_tr, f_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        pre = build_preprocessor(num_cols, cat_cols)
        X_tr = pre.fit_transform(f_tr[all_features])
        X_val = pre.transform(f_val[all_features])
        
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
        clf.fit(X_tr, f_tr["target"].values)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
    
    print(f"OOF Probabilities generated. Overall OOF ROC-AUC: {compute_binary_metrics(y_train, oof_probs, 0.5)['roc_auc']:.4f}", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. Select Candidate Decision Thresholds on OOF Validation Data ONLY
    # -------------------------------------------------------------------------
    print("\nStep 2: Selecting Candidate Decision Thresholds on OOF Data...", flush=True)
    candidate_thresholds = np.linspace(0.01, 0.99, 100)
    best_f1 = -1.0
    tau_max_f1 = 0.50
    best_youden = -1.0
    tau_youden = 0.50
    tau_recall_80 = 0.50
    closest_recall_diff = 999.0
    
    tau_search_records = []
    for tau in candidate_thresholds:
        m = compute_binary_metrics(y_train, oof_probs, threshold=tau)
        tau_search_records.append({
            "threshold": tau,
            "f1": m["f1"],
            "recall": m["recall"],
            "precision": m["precision"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "youden_j": m["recall"] + m["specificity"] - 1.0
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            tau_max_f1 = tau
        youden = m["recall"] + m["specificity"] - 1.0
        if youden > best_youden:
            best_youden = youden
            tau_youden = tau
        if abs(m["recall"] - 0.80) < closest_recall_diff:
            closest_recall_diff = abs(m["recall"] - 0.80)
            tau_recall_80 = tau
            
    print(f"  Frozen Candidate Thresholds (Tuned on OOF Only):")
    print(f"    1. Default:             tau = 0.5000")
    print(f"    2. Max F1:              tau = {tau_max_f1:.4f} (OOF F1 = {best_f1:.4f})")
    print(f"    3. Target Recall (80%): tau = {tau_recall_80:.4f}")
    print(f"    4. Youden's J:          tau = {tau_youden:.4f} (OOF J = {best_youden:.4f})")
    
    # -------------------------------------------------------------------------
    # 3. Fit Calibration Models Strictly on OOF Probabilities
    # -------------------------------------------------------------------------
    print("\nStep 3: Fitting Probability Calibration on OOF Data...", flush=True)
    # Platt Scaling (Sigmoid)
    platt = LogisticRegression(solver="lbfgs", max_iter=500, random_state=seed)
    platt.fit(oof_probs.reshape(-1, 1), y_train)
    
    # Isotonic Regression
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_probs, y_train)
    
    # -------------------------------------------------------------------------
    # 4. Fit Base Model on Full Training Data & Evaluate on Untouched Test Set
    # -------------------------------------------------------------------------
    print("\nStep 4: Fitting on Full Training Data & Evaluating on Untouched Test Set...", flush=True)
    full_pre = build_preprocessor(num_cols, cat_cols)
    X_train_full = full_pre.fit_transform(train_df[all_features])
    X_test_full = full_pre.transform(test_df[all_features])
    
    full_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    full_model.fit(X_train_full, y_train)
    
    # Predict uncalibrated test probabilities
    y_test_uncal = full_model.predict_proba(X_test_full)[:, 1]
    
    # Apply frozen calibration transforms
    y_test_platt = platt.predict_proba(y_test_uncal.reshape(-1, 1))[:, 1]
    y_test_iso = iso.predict(y_test_uncal)
    
    # Calibration evaluation
    cal_results = []
    for cal_name, probs in [
        ("Uncalibrated", y_test_uncal),
        ("Sigmoid_Platt", y_test_platt),
        ("Isotonic", y_test_iso)
    ]:
        ece, bin_stats = compute_ece(y_test, probs, n_bins=10, strategy="uniform")
        m = compute_binary_metrics(y_test, probs, threshold=0.5)
        cal_results.append({
            "calibration_method": cal_name,
            "brier_score": m["brier_score"],
            "log_loss": m["log_loss"],
            "ece": ece,
            "roc_auc": m["roc_auc"],
            "pr_auc": m["pr_auc"]
        })
        print(f"  [{cal_name}] Brier: {m['brier_score']:.5f} | LogLoss: {m['log_loss']:.5f} | ECE: {ece:.4f} | ROC-AUC: {m['roc_auc']:.4f}")
    
    cal_df = pd.DataFrame(cal_results)
    cal_df.to_csv("results/tables/calibration_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 5. Evaluate Frozen Thresholds on Independent Test Set
    # -------------------------------------------------------------------------
    print("\nStep 5: Evaluating Frozen Thresholds on Untouched Test Set...", flush=True)
    threshold_dict = {
        "Default_0.50": 0.50,
        "Max_F1": tau_max_f1,
        "Target_Recall_80": tau_recall_80,
        "Youden_J": tau_youden
    }
    
    thresh_eval_results = []
    for t_name, tau in threshold_dict.items():
        # Evaluate on uncalibrated probabilities
        m_uncal = compute_binary_metrics(y_test, y_test_uncal, threshold=tau)
        ci = compute_bootstrap_ci(y_test, y_test_uncal, threshold=tau, n_bootstraps=500, seed=seed)
        
        row = {
            "threshold_name": t_name,
            "frozen_threshold_value": float(tau),
            "probability_type": "Uncalibrated",
            "accuracy": m_uncal["accuracy"],
            "precision": m_uncal["precision"],
            "recall": m_uncal["recall"],
            "recall_ci_lower": ci["recall"]["ci_lower"],
            "recall_ci_upper": ci["recall"]["ci_upper"],
            "specificity": m_uncal["specificity"],
            "specificity_ci_lower": ci["specificity"]["ci_lower"],
            "specificity_ci_upper": ci["specificity"]["ci_upper"],
            "f1": m_uncal["f1"],
            "f1_ci_lower": ci["f1"]["ci_lower"],
            "f1_ci_upper": ci["f1"]["ci_upper"],
            "balanced_accuracy": m_uncal["balanced_accuracy"],
            "mcc": m_uncal["mcc"],
            "tp": m_uncal["tp"],
            "fp": m_uncal["fp"],
            "tn": m_uncal["tn"],
            "fn": m_uncal["fn"]
        }
        thresh_eval_results.append(row)
        print(f"  [{t_name} (tau={tau:.4f})] Recall: {m_uncal['recall']:.4f} [{ci['recall']['ci_lower']:.4f}, {ci['recall']['ci_upper']:.4f}] | "
              f"Spec: {m_uncal['specificity']:.4f} | F1: {m_uncal['f1']:.4f} | Acc: {m_uncal['accuracy']:.4f} | "
              f"TP={m_uncal['tp']}, FP={m_uncal['fp']}, TN={m_uncal['tn']}, FN={m_uncal['fn']}")
    
    thresh_df = pd.DataFrame(thresh_eval_results)
    thresh_df.to_csv("results/tables/threshold_comparison.csv", index=False)
    
    # -------------------------------------------------------------------------
    # 6. Figures: Calibration Curves, Threshold Curves, Confusion Matrices
    # -------------------------------------------------------------------------
    print("\nStep 6: Generating Figures...", flush=True)
    # Figure 1: Calibration Curves
    plt.figure(figsize=(8, 7))
    for cal_name, probs, col in [
        ("Uncalibrated", y_test_uncal, "steelblue"),
        ("Sigmoid (Platt)", y_test_platt, "forestgreen"),
        ("Isotonic", y_test_iso, "darkorange")
    ]:
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="uniform")
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label=cal_name, color=col)
    
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability", fontsize=12)
    plt.ylabel("Observed Fraction of Positives", fontsize=12)
    plt.title("Reliability Diagram (Calibration Curves on Test Set)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("results/figures/calibration_curves.png", dpi=300)
    plt.close()
    
    # Figure 2: Threshold vs Metrics Curve
    tau_df = pd.DataFrame(tau_search_records)
    plt.figure(figsize=(10, 6))
    plt.plot(tau_df["threshold"], tau_df["recall"], label="Recall (Sensitivity)", color="crimson", linewidth=2)
    plt.plot(tau_df["threshold"], tau_df["specificity"], label="Specificity", color="navy", linewidth=2)
    plt.plot(tau_df["threshold"], tau_df["f1"], label="F1-Score", color="forestgreen", linewidth=2)
    plt.plot(tau_df["threshold"], tau_df["precision"], label="Precision (PPV)", color="darkorange", linewidth=2)
    plt.plot(tau_df["threshold"], tau_df["balanced_accuracy"], label="Balanced Accuracy", color="purple", linestyle="--", linewidth=2)
    
    plt.axvline(0.50, color="gray", linestyle=":", label="Default tau=0.50")
    plt.axvline(tau_max_f1, color="forestgreen", linestyle="--", label=f"Max F1 tau={tau_max_f1:.3f}")
    plt.axvline(tau_youden, color="navy", linestyle="-.", label=f"Youden tau={tau_youden:.3f}")
    
    plt.xlabel("Classification Decision Threshold (tau)", fontsize=12)
    plt.ylabel("Metric Score", fontsize=12)
    plt.title("Effect of Decision Threshold on Operational Performance", fontsize=14, fontweight="bold")
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig("results/figures/threshold_metric_curves.png", dpi=300)
    plt.close()
    
    # Figure 3: Confusion Matrices for Key Frozen Thresholds
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cm_configs = [
        ("Default (0.50)", 0.50),
        (f"Max F1 ({tau_max_f1:.3f})", tau_max_f1),
        (f"Youden J ({tau_youden:.3f})", tau_youden)
    ]
    for i, (title, tau) in enumerate(cm_configs):
        y_pred = (y_test_uncal >= tau).astype(int)
        cm = [[np.sum((y_test==0)&(y_pred==0)), np.sum((y_test==0)&(y_pred==1))],
              [np.sum((y_test==1)&(y_pred==0)), np.sum((y_test==1)&(y_pred==1))]]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["No (0)", "Yes (1)"], yticklabels=["No (0)", "Yes (1)"])
        axes[i].set_title(title, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Predicted Heart Disease")
        axes[i].set_ylabel("Actual Heart Disease")
    
    plt.tight_layout()
    plt.savefig("results/figures/final_confusion_matrices.png", dpi=300)
    plt.close()
    
    print("Saved all tables and figures for Experiment 4.", flush=True)

if __name__ == "__main__":
    run_threshold_and_calibration()
