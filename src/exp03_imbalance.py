"""
Experiment 3: Class-Imbalance Handling Comparison
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
from sklearn.metrics import precision_recall_curve

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier

from src.data_loader import load_clean_data, build_preprocessor, load_config
from src.metrics import compute_binary_metrics, compute_bootstrap_ci

def run_imbalance_comparison():
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 3: CLASS-IMBALANCE HANDLING EVALUATION", flush=True)
    print("=" * 70, flush=True)
    
    df, train_df, test_df, config = load_clean_data()
    seed = config["project"]["random_seed"]
    all_features = config["feature_groups"]["group_d_all_features"]
    num_cols = [c for c in config["features"]["numerical"] if c in all_features]
    cat_cols = [c for c in config["features"]["categorical"] if c in all_features]
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Preprocessor for standard encoded representations
    pre = build_preprocessor(num_cols, cat_cols)
    X_train_encoded = pre.fit_transform(train_df[all_features])
    X_test_encoded = pre.transform(test_df[all_features])
    
    # Calculate scale_pos_weight for XGBoost
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos = float(n_neg / n_pos)
    
    # Define Resampling Methods to evaluate
    methods = [
        {"name": "A_No_Resampling", "type": "baseline"},
        {"name": "B_Class_Weighting", "type": "weighting"},
        {"name": "C_SMOTE_0.25", "type": "smote", "ratio": 0.25},
        {"name": "C_SMOTE_0.50", "type": "smote", "ratio": 0.50},
        {"name": "C_SMOTE_0.75", "type": "smote", "ratio": 0.75},
        {"name": "C_SMOTE_1.00", "type": "smote", "ratio": 1.00},
        {"name": "D_SMOTENC", "type": "smotenc", "ratio": 0.50},
        {"name": "E_Borderline_SMOTE", "type": "borderline", "ratio": 0.50},
        {"name": "F_SMOTE_Tomek", "type": "tomek", "ratio": 0.50},
        {"name": "G_Random_Oversampling", "type": "random_over", "ratio": 0.50},
        {"name": "H_Balanced_Random_Forest", "type": "brf"}
    ]
    
    fold_records = []
    summary_records = []
    pr_curve_data = {}
    
    # We use XGBoost and LogisticRegression across methods, plus BalancedRandomForest for Method H
    for m_cfg in methods:
        m_name = m_cfg["name"]
        m_type = m_cfg["type"]
        print(f"\nEvaluating Imbalance Method: {m_name}...", flush=True)
        
        # Determine model to evaluate
        if m_type == "brf":
            eval_models = [("BalancedRandomForest", BalancedRandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1))]
        elif m_type == "weighting":
            eval_models = [
                ("LogisticRegression_Weighted", LogisticRegression(class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=seed)),
                ("XGBoost_Weighted", XGBClassifier(scale_pos_weight=scale_pos, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
            ]
        else:
            eval_models = [
                ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)),
                ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed))
            ]
            
        for model_name, model_template in eval_models:
            t0 = time.time()
            fold_metrics_list = []
            resampled_sizes = []
            
            for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
                f_tr_df = train_df.iloc[tr_idx]
                f_val_df = train_df.iloc[val_idx]
                
                # Preprocess fold data strictly within fold
                fold_pre = build_preprocessor(num_cols, cat_cols)
                X_tr = fold_pre.fit_transform(f_tr_df[all_features])
                y_tr = f_tr_df["target"].values
                X_val = fold_pre.transform(f_val_df[all_features])
                y_val = f_val_df["target"].values
                
                # Apply Resampling method strictly to fold training data
                if m_type == "baseline" or m_type == "weighting" or m_type == "brf":
                    X_tr_res, y_tr_res = X_tr, y_tr
                elif m_type == "smote":
                    sampler = SMOTE(sampling_strategy=m_cfg["ratio"], random_state=seed)
                    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
                elif m_type == "borderline":
                    sampler = BorderlineSMOTE(sampling_strategy=m_cfg["ratio"], random_state=seed)
                    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
                elif m_type == "random_over":
                    sampler = RandomOverSampler(sampling_strategy=m_cfg["ratio"], random_state=seed)
                    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
                elif m_type == "tomek":
                    sampler = SMOTETomek(sampling_strategy=m_cfg["ratio"], random_state=seed, n_jobs=-1)
                    X_tr_res, y_tr_res = sampler.fit_resample(X_tr, y_tr)
                elif m_type == "smotenc":
                    # SMOTENC on unencoded categorical representation
                    cat_indices = [f_tr_df[all_features].columns.get_loc(c) for c in cat_cols]
                    sampler = SMOTENC(categorical_features=cat_indices, sampling_strategy=m_cfg["ratio"], random_state=seed)
                    X_raw_res, y_tr_res = sampler.fit_resample(f_tr_df[all_features], y_tr)
                    # Now encode resampled data using preprocessor fit on raw resampled
                    fold_pre_nc = build_preprocessor(num_cols, cat_cols)
                    X_tr_res = fold_pre_nc.fit_transform(X_raw_res)
                    X_val = fold_pre_nc.transform(f_val_df[all_features])
                
                resampled_sizes.append(len(y_tr_res))
                
                # Fit model
                if "XGB" in model_name:
                    clf = XGBClassifier(scale_pos_weight=scale_pos if "Weighted" in model_name else 1.0,
                                        n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
                elif "Logistic" in model_name:
                    clf = LogisticRegression(class_weight="balanced" if "Weighted" in model_name else None,
                                             max_iter=1000, solver="lbfgs", random_state=seed)
                elif "BalancedRandomForest" in model_name:
                    clf = BalancedRandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
                
                clf.fit(X_tr_res, y_tr_res)
                y_val_prob = clf.predict_proba(X_val)[:, 1]
                
                m = compute_binary_metrics(y_val, y_val_prob, threshold=0.5)
                m["fold"] = fold
                m["method"] = m_name
                m["model"] = model_name
                m["train_size_before"] = len(y_tr)
                m["train_size_after"] = len(y_tr_res)
                m["resampled_pos_ratio"] = float(np.mean(y_tr_res))
                fold_records.append(m)
                fold_metrics_list.append(m)
                
            elapsed = time.time() - t0
            
            # Now fit on full training set and evaluate on untouched test set
            full_pre = build_preprocessor(num_cols, cat_cols)
            X_tr_full = full_pre.fit_transform(train_df[all_features])
            y_tr_full = train_df["target"].values
            X_te_full = full_pre.transform(test_df[all_features])
            
            if m_type == "baseline" or m_type == "weighting" or m_type == "brf":
                X_tr_full_res, y_tr_full_res = X_tr_full, y_tr_full
            elif m_type == "smote":
                s = SMOTE(sampling_strategy=m_cfg["ratio"], random_state=seed)
                X_tr_full_res, y_tr_full_res = s.fit_resample(X_tr_full, y_tr_full)
            elif m_type == "borderline":
                s = BorderlineSMOTE(sampling_strategy=m_cfg["ratio"], random_state=seed)
                X_tr_full_res, y_tr_full_res = s.fit_resample(X_tr_full, y_tr_full)
            elif m_type == "random_over":
                s = RandomOverSampler(sampling_strategy=m_cfg["ratio"], random_state=seed)
                X_tr_full_res, y_tr_full_res = s.fit_resample(X_tr_full, y_tr_full)
            elif m_type == "tomek":
                s = SMOTETomek(sampling_strategy=m_cfg["ratio"], random_state=seed, n_jobs=-1)
                X_tr_full_res, y_tr_full_res = s.fit_resample(X_tr_full, y_tr_full)
            elif m_type == "smotenc":
                cat_indices = [train_df[all_features].columns.get_loc(c) for c in cat_cols]
                s = SMOTENC(categorical_features=cat_indices, sampling_strategy=m_cfg["ratio"], random_state=seed)
                X_raw_full_res, y_tr_full_res = s.fit_resample(train_df[all_features], y_tr_full)
                full_pre_nc = build_preprocessor(num_cols, cat_cols)
                X_tr_full_res = full_pre_nc.fit_transform(X_raw_full_res)
                X_te_full = full_pre_nc.transform(test_df[all_features])
                
            if "XGB" in model_name:
                final_clf = XGBClassifier(scale_pos_weight=scale_pos if "Weighted" in model_name else 1.0,
                                          n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
            elif "Logistic" in model_name:
                final_clf = LogisticRegression(class_weight="balanced" if "Weighted" in model_name else None,
                                               max_iter=1000, solver="lbfgs", random_state=seed)
            elif "BalancedRandomForest" in model_name:
                final_clf = BalancedRandomForestClassifier(n_estimators=100, max_depth=12, random_state=seed, n_jobs=-1)
                
            final_clf.fit(X_tr_full_res, y_tr_full_res)
            y_test_prob = final_clf.predict_proba(X_te_full)[:, 1]
            test_m = compute_binary_metrics(y_test, y_test_prob, threshold=0.5)
            ci = compute_bootstrap_ci(y_test, y_test_prob, threshold=0.5, n_bootstraps=500, seed=seed)
            
            # Save PR curve
            prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
            pr_curve_data[f"{m_name}_{model_name}"] = (rec, prec, test_m["pr_auc"])
            
            print(f"  [{model_name}] Test ROC-AUC={test_m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}] | "
                  f"Test PR-AUC={test_m['pr_auc']:.4f} | Recall={test_m['recall']:.4f} | Spec={test_m['specificity']:.4f} | "
                  f"Brier={test_m['brier_score']:.4f} | Time={elapsed:.1f}s", flush=True)
            
            summary_records.append({
                "imbalance_method": m_name,
                "model": model_name,
                "resampled_train_size_mean": float(np.mean(resampled_sizes)),
                "cv_roc_auc_mean": float(np.mean([x["roc_auc"] for x in fold_metrics_list])),
                "cv_roc_auc_std": float(np.std([x["roc_auc"] for x in fold_metrics_list])),
                "cv_pr_auc_mean": float(np.mean([x["pr_auc"] for x in fold_metrics_list])),
                "cv_pr_auc_std": float(np.std([x["pr_auc"] for x in fold_metrics_list])),
                "cv_f1_mean": float(np.mean([x["f1"] for x in fold_metrics_list])),
                "cv_recall_mean": float(np.mean([x["recall"] for x in fold_metrics_list])),
                "cv_specificity_mean": float(np.mean([x["specificity"] for x in fold_metrics_list])),
                "cv_bal_acc_mean": float(np.mean([x["balanced_accuracy"] for x in fold_metrics_list])),
                "cv_mcc_mean": float(np.mean([x["mcc"] for x in fold_metrics_list])),
                "test_roc_auc": test_m["roc_auc"],
                "test_roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
                "test_roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
                "test_pr_auc": test_m["pr_auc"],
                "test_pr_auc_ci_lower": ci["pr_auc"]["ci_lower"],
                "test_pr_auc_ci_upper": ci["pr_auc"]["ci_upper"],
                "test_brier": test_m["brier_score"],
                "test_accuracy": test_m["accuracy"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_specificity": test_m["specificity"],
                "test_f1": test_m["f1"],
                "test_balanced_acc": test_m["balanced_accuracy"],
                "test_mcc": test_m["mcc"],
                "runtime_seconds": float(elapsed)
            })
            
    sum_df = pd.DataFrame(summary_records)
    sum_df.to_csv("results/tables/imbalance_comparison.csv", index=False)
    
    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("results/tables/imbalance_fold_results.csv", index=False)
    print("\nSaved results/tables/imbalance_comparison.csv and results/tables/imbalance_fold_results.csv", flush=True)
    
    # Generate Figures
    # Figure 1: Imbalance Metric Comparison (ROC-AUC vs PR-AUC vs Brier Score)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: PR-AUC across methods
    sns.barplot(data=sum_df, x="imbalance_method", y="test_pr_auc", hue="model", ax=axes[0], palette="Spectral")
    axes[0].set_title("Test PR-AUC (Average Precision) by Imbalance Method", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("PR-AUC", fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].legend(loc="upper right")
    
    # Subplot 2: Probability Calibration Disruption (Brier Score)
    sns.barplot(data=sum_df, x="imbalance_method", y="test_brier", hue="model", ax=axes[1], palette="Spectral")
    axes[1].set_title("Probability Calibration Distortion (Brier Score: Lower is Better)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Brier Score Loss", fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("results/figures/imbalance_metric_comparison.png", dpi=300)
    plt.close()
    
    # Figure 2: Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for k, (rec, prec, auc_val) in pr_curve_data.items():
        if "XGBoost" in k or "BalancedRandomForest" in k:
            plt.plot(rec, prec, label=f"{k} (AUC={auc_val:.3f})", linewidth=1.8)
    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title("Precision-Recall Curves across Imbalance Handling Techniques", fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("results/figures/precision_recall_curves.png", dpi=300)
    plt.close()
    print("Saved results/figures/imbalance_metric_comparison.png and precision_recall_curves.png", flush=True)

if __name__ == "__main__":
    run_imbalance_comparison()
