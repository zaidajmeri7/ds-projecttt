"""
Experiment 1: Demographic Shortcut Analysis & Feature Ablation
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
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data_loader import load_clean_data, build_preprocessor, load_config
from src.metrics import compute_binary_metrics, compute_bootstrap_ci

def get_models(seed=42):
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    }

def run_demographic_ablation():
    print("=" * 70)
    print("STARTING EXPERIMENT 1: DEMOGRAPHIC SHORTCUT & FEATURE ABLATION")
    print("=" * 70)
    
    df, train_df, test_df, config = load_clean_data()
    seed = config["project"]["random_seed"]
    
    # Define feature groups
    feature_groups = {
        "Group_A_Demographics": config["feature_groups"]["group_a_demographics"],
        "Group_A_Age_Sex": config["feature_groups"]["group_a_age_sex_only"],
        "Group_B_Lifestyle": config["feature_groups"]["group_b_lifestyle"],
        "Group_B_Lifestyle_GenHealth": config["feature_groups"]["group_b_lifestyle_with_genhealth"],
        "Group_C_HealthStatus": config["feature_groups"]["group_c_health_status"],
        "Group_D_AllFeatures": config["feature_groups"]["group_d_all_features"],
        "Group_E_NoDemographics": config["feature_groups"]["group_e_no_demographics"]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_train = train_df["target"].values
    y_test = test_df["target"].values
    
    all_results = []
    roc_curves_data = {}
    
    for group_name, features in feature_groups.items():
        print(f"\nEvaluating Feature Group: {group_name} ({len(features)} features: {features})")
        num_cols = [c for c in config["features"]["numerical"] if c in features]
        cat_cols = [c for c in config["features"]["categorical"] if c in features]
        
        models = get_models(seed=seed)
        
        for model_name, model in models.items():
            t0 = time.time()
            cv_roc_aucs = []
            cv_pr_aucs = []
            cv_briers = []
            cv_f1s = []
            cv_recalls = []
            cv_specificities = []
            cv_bal_accs = []
            cv_mccs = []
            
            # 5-fold cross-validation on training data
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
                fold_train = train_df.iloc[train_idx]
                fold_val = train_df.iloc[val_idx]
                
                # Fit preprocessor strictly on fold training data
                preprocessor = build_preprocessor(num_cols, cat_cols)
                X_tr = preprocessor.fit_transform(fold_train[features])
                X_val = preprocessor.transform(fold_val[features])
                
                # Clone/reset model
                fold_model = get_models(seed=seed)[model_name]
                fold_model.fit(X_tr, fold_train["target"].values)
                
                y_val_prob = fold_model.predict_proba(X_val)[:, 1]
                m = compute_binary_metrics(fold_val["target"].values, y_val_prob, threshold=0.5)
                
                cv_roc_aucs.append(m["roc_auc"])
                cv_pr_aucs.append(m["pr_auc"])
                cv_briers.append(m["brier_score"])
                cv_f1s.append(m["f1"])
                cv_recalls.append(m["recall"])
                cv_specificities.append(m["specificity"])
                cv_bal_accs.append(m["balanced_accuracy"])
                cv_mccs.append(m["mcc"])
            
            # Fit on full training data and evaluate on untouched test set
            full_preprocessor = build_preprocessor(num_cols, cat_cols)
            X_train_full = full_preprocessor.fit_transform(train_df[features])
            X_test_full = full_preprocessor.transform(test_df[features])
            
            full_model = get_models(seed=seed)[model_name]
            full_model.fit(X_train_full, y_train)
            y_test_prob = full_model.predict_proba(X_test_full)[:, 1]
            
            test_metrics = compute_binary_metrics(y_test, y_test_prob, threshold=0.5)
            
            # Bootstrap 95% CIs on test set (500 iterations for speed and robustness)
            ci = compute_bootstrap_ci(y_test, y_test_prob, threshold=0.5, n_bootstraps=500, seed=seed)
            
            elapsed = time.time() - t0
            print(f"  [{model_name}] CV ROC-AUC: {np.mean(cv_roc_aucs):.4f} +/- {np.std(cv_roc_aucs):.4f} | "
                  f"Test ROC-AUC: {test_metrics['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}] | "
                  f"Test PR-AUC: {test_metrics['pr_auc']:.4f} | Time: {elapsed:.1f}s")
            
            res = {
                "feature_group": group_name,
                "n_features": len(features),
                "model": model_name,
                "cv_roc_auc_mean": float(np.mean(cv_roc_aucs)),
                "cv_roc_auc_std": float(np.std(cv_roc_aucs)),
                "cv_pr_auc_mean": float(np.mean(cv_pr_aucs)),
                "cv_pr_auc_std": float(np.std(cv_pr_aucs)),
                "cv_brier_mean": float(np.mean(cv_briers)),
                "cv_f1_mean": float(np.mean(cv_f1s)),
                "cv_recall_mean": float(np.mean(cv_recalls)),
                "cv_specificity_mean": float(np.mean(cv_specificities)),
                "cv_bal_acc_mean": float(np.mean(cv_bal_accs)),
                "cv_mcc_mean": float(np.mean(cv_mccs)),
                "test_roc_auc": test_metrics["roc_auc"],
                "test_roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
                "test_roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
                "test_pr_auc": test_metrics["pr_auc"],
                "test_pr_auc_ci_lower": ci["pr_auc"]["ci_lower"],
                "test_pr_auc_ci_upper": ci["pr_auc"]["ci_upper"],
                "test_brier": test_metrics["brier_score"],
                "test_log_loss": test_metrics["log_loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_specificity": test_metrics["specificity"],
                "test_f1": test_metrics["f1"],
                "test_balanced_acc": test_metrics["balanced_accuracy"],
                "test_mcc": test_metrics["mcc"],
                "test_tp": test_metrics["tp"],
                "test_fp": test_metrics["fp"],
                "test_tn": test_metrics["tn"],
                "test_fn": test_metrics["fn"],
                "runtime_seconds": float(elapsed)
            }
            all_results.append(res)
            
            # Store ROC curve data for key models
            if model_name == "LogisticRegression" or model_name == "XGBoost":
                fpr, tpr, _ = roc_curve(y_test, y_test_prob)
                roc_curves_data[f"{group_name}_{model_name}"] = (fpr, tpr, test_metrics["roc_auc"])
    
    res_df = pd.DataFrame(all_results)
    
    # Calculate Delta AUC and Delta AP relative to Group A Demographics for each model
    demo_ref = res_df[res_df["feature_group"] == "Group_A_Demographics"].set_index("model")
    
    res_df["delta_auc_vs_demographics"] = res_df.apply(
        lambda r: r["test_roc_auc"] - demo_ref.loc[r["model"], "test_roc_auc"], axis=1
    )
    res_df["delta_pr_auc_vs_demographics"] = res_df.apply(
        lambda r: r["test_pr_auc"] - demo_ref.loc[r["model"], "test_pr_auc"], axis=1
    )
    
    # Save tables
    os.makedirs("results/tables", exist_ok=True)
    res_df.to_csv("results/tables/feature_group_comparison.csv", index=False)
    
    # Demographic ablation table focusing on key comparisons
    ablation_df = res_df[["feature_group", "model", "test_roc_auc", "test_pr_auc", 
                          "delta_auc_vs_demographics", "delta_pr_auc_vs_demographics",
                          "test_recall", "test_specificity", "test_f1", "test_brier"]]
    ablation_df.to_csv("results/tables/demographic_ablation.csv", index=False)
    print("\nSaved results/tables/feature_group_comparison.csv and results/tables/demographic_ablation.csv")
    
    # Generate Figures
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Barplot of Test ROC-AUC by Feature Group and Model
    sns.barplot(data=res_df, x="feature_group", y="test_roc_auc", hue="model", ax=axes[0], palette="Set2")
    axes[0].set_title("Test ROC-AUC across Feature Groups", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC", fontsize=12)
    axes[0].set_xlabel("Feature Group", fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].set_ylim(0.70, 0.86)
    axes[0].axhline(0.7596, color="red", linestyle="--", alpha=0.7, label="PDF Age+Sex Claim (0.7596)")
    axes[0].legend(loc="lower right")
    
    # Subplot 2: Delta AUC vs Demographics
    sns.barplot(data=res_df, x="feature_group", y="delta_auc_vs_demographics", hue="model", ax=axes[1], palette="Set2")
    axes[1].set_title("Incremental Clinical Value (Delta AUC vs Demographics)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Delta ROC-AUC", fontsize=12)
    axes[1].set_xlabel("Feature Group", fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    axes[1].legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig("results/figures/feature_group_comparison.png", dpi=300)
    plt.close()
    print("Saved results/figures/feature_group_comparison.png")
    
    return res_df

if __name__ == "__main__":
    run_demographic_ablation()
