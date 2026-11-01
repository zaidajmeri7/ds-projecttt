"""
Experiment 2: Preprocessing and Resampling Leakage Trap Demonstration
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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler

from src.data_loader import load_clean_data, build_preprocessor, load_config
from src.metrics import compute_binary_metrics, compute_bootstrap_ci

def run_leakage_audit():
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 2: PREPROCESSING AND RESAMPLING LEAKAGE AUDIT", flush=True)
    print("=" * 70, flush=True)
    
    config = load_config()
    raw_path = config["data"]["raw_path"]
    df = pd.read_csv(raw_path)
    df["target"] = (df[config["data"]["target_column"]] == config["data"]["positive_label"]).astype(int)
    
    seed = config["project"]["random_seed"]
    all_features = config["feature_groups"]["group_d_all_features"]
    num_cols = [c for c in config["features"]["numerical"] if c in all_features]
    cat_cols = [c for c in config["features"]["categorical"] if c in all_features]
    
    log_file = "results/logs/leakage_audit.log"
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    log_lines = []
    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)
    
    log(f"Dataset records: {len(df)}")
    log(f"Original class balance: No={int((df['target']==0).sum())}, Yes={int((df['target']==1).sum())} ({df['target'].mean()*100:.2f}% positive)")
    
    results = []
    
    # -------------------------------------------------------------------------
    # PIPELINE 1: Clean Isolated Baseline (No Resampling)
    # Preprocessor fit on train only, test untouched.
    # -------------------------------------------------------------------------
    log("\n[PIPELINE 1] Clean Isolated Baseline (No Resampling)...")
    train_df, test_df = train_test_split(df, test_size=0.20, stratify=df["target"], random_state=seed)
    
    pre = build_preprocessor(num_cols, cat_cols)
    X_train = pre.fit_transform(train_df[all_features])
    y_train = train_df["target"].values
    X_test = pre.transform(test_df[all_features])
    y_test = test_df["target"].values
    
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
    ]:
        t0 = time.time()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        ci = compute_bootstrap_ci(y_test, y_prob, threshold=0.5, n_bootstraps=500, seed=seed)
        elapsed = time.time() - t0
        
        log(f"  {model_name}: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        results.append({
            "pipeline": "1_Clean_Baseline_NoResample",
            "model": model_name,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_pos_pct": float(np.mean(y_train) * 100),
            "test_pos_pct": float(np.mean(y_test) * 100),
            "synthetic_samples_in_test": 0,
            "duplicate_samples_in_test": 0,
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "brier_score": m["brier_score"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "runtime_s": float(elapsed)
        })
    
    # -------------------------------------------------------------------------
    # PIPELINE 2: Clean Isolated SMOTE
    # Split first -> fit preprocessor on train -> SMOTE on train only -> test untouched
    # -------------------------------------------------------------------------
    log("\n[PIPELINE 2] Clean Isolated SMOTE (Resampling on Train Only)...")
    smote = SMOTE(random_state=seed)
    t_sm = time.time()
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    log(f"  SMOTE on train completed in {time.time()-t_sm:.2f}s: Train shape grew from {X_train.shape} to {X_train_sm.shape}")
    
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
    ]:
        t0 = time.time()
        model.fit(X_train_sm, y_train_sm)
        y_prob = model.predict_proba(X_test)[:, 1]
        m = compute_binary_metrics(y_test, y_prob, threshold=0.5)
        ci = compute_bootstrap_ci(y_test, y_prob, threshold=0.5, n_bootstraps=500, seed=seed)
        elapsed = time.time() - t0
        
        log(f"  {model_name}: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        results.append({
            "pipeline": "2_Clean_SMOTE_TrainOnly",
            "model": model_name,
            "train_size": len(y_train_sm),
            "test_size": len(y_test),
            "train_pos_pct": float(np.mean(y_train_sm) * 100),
            "test_pos_pct": float(np.mean(y_test) * 100),
            "synthetic_samples_in_test": 0,
            "duplicate_samples_in_test": 0,
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "brier_score": m["brier_score"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "runtime_s": float(elapsed)
        })
    
    # -------------------------------------------------------------------------
    # PIPELINE 3: Deliberate SMOTE Resampling Leakage Before Train-Test Split
    # Global preprocessor -> SMOTE on entire dataset -> then 80:20 split!
    # -------------------------------------------------------------------------
    log("\n[PIPELINE 3] Deliberate Resampling Leakage: SMOTE Before Split...")
    pre_global = build_preprocessor(num_cols, cat_cols)
    X_all = pre_global.fit_transform(df[all_features])
    y_all = df["target"].values
    
    log(f"  Preprocessed full dataset: {X_all.shape}. Applying SMOTE to entire dataset before split...")
    smote_leaked = SMOTE(random_state=seed)
    X_leaked_all, y_leaked_all = smote_leaked.fit_resample(X_all, y_all)
    log(f"  Oversampled full dataset shape: {X_leaked_all.shape} (balanced: {int((y_leaked_all==1).sum())} positive, {int((y_leaked_all==0).sum())} negative)")
    
    X_tr_leak, X_te_leak, y_tr_leak, y_te_leak = train_test_split(
        X_leaked_all, y_leaked_all, test_size=0.20, stratify=y_leaked_all, random_state=seed
    )
    
    synthetic_test_count = int(np.sum(y_te_leak == 1) - (np.sum(y_all == 1) * 0.20))
    log(f"  Leaked Train size: {len(y_tr_leak)}, Leaked Test size: {len(y_te_leak)}")
    log(f"  Synthetic/duplicated minority cases bleeding into test partition: ~{synthetic_test_count}")
    
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
    ]:
        t0 = time.time()
        model.fit(X_tr_leak, y_tr_leak)
        y_prob_leak = model.predict_proba(X_te_leak)[:, 1]
        m = compute_binary_metrics(y_te_leak, y_prob_leak, threshold=0.5)
        ci = compute_bootstrap_ci(y_te_leak, y_prob_leak, threshold=0.5, n_bootstraps=500, seed=seed)
        elapsed = time.time() - t0
        
        log(f"  {model_name}: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        results.append({
            "pipeline": "3_Leaked_SMOTE_Before_Split",
            "model": model_name,
            "train_size": len(y_tr_leak),
            "test_size": len(y_te_leak),
            "train_pos_pct": float(np.mean(y_tr_leak) * 100),
            "test_pos_pct": float(np.mean(y_te_leak) * 100),
            "synthetic_samples_in_test": int(synthetic_test_count),
            "duplicate_samples_in_test": -1, # synthetic vectors interpolated across partitions
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "brier_score": m["brier_score"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "runtime_s": float(elapsed)
        })
    
    # -------------------------------------------------------------------------
    # PIPELINE 4: Random Oversampling Leakage Before Train-Test Split (Exact Duplication)
    # -------------------------------------------------------------------------
    log("\n[PIPELINE 4] Deliberate Random Oversampling (Exact Duplication) Leakage...")
    ros = RandomOverSampler(random_state=seed)
    indices = np.arange(len(df)).reshape(-1, 1)
    idx_resampled, y_resampled = ros.fit_resample(indices, df["target"].values)
    
    train_idx, test_idx = train_test_split(
        np.arange(len(idx_resampled)), test_size=0.20, stratify=y_resampled, random_state=seed
    )
    
    train_orig_indices = set(idx_resampled[train_idx].ravel())
    test_orig_indices = set(idx_resampled[test_idx].ravel())
    leaked_duplicate_count = len(train_orig_indices.intersection(test_orig_indices))
    log(f"  Exact patient records appearing in BOTH train and test partitions: {leaked_duplicate_count}")
    
    X_tr_ros = X_all[idx_resampled[train_idx].ravel()]
    y_tr_ros = y_resampled[train_idx]
    X_te_ros = X_all[idx_resampled[test_idx].ravel()]
    y_te_ros = y_resampled[test_idx]
    
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
    ]:
        t0 = time.time()
        model.fit(X_tr_ros, y_tr_ros)
        y_prob_ros = model.predict_proba(X_te_ros)[:, 1]
        m = compute_binary_metrics(y_te_ros, y_prob_ros, threshold=0.5)
        ci = compute_bootstrap_ci(y_te_ros, y_prob_ros, threshold=0.5, n_bootstraps=500, seed=seed)
        elapsed = time.time() - t0
        
        log(f"  {model_name}: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        results.append({
            "pipeline": "4_Leaked_RandomOverSample_Before_Split",
            "model": model_name,
            "train_size": len(y_tr_ros),
            "test_size": len(y_te_ros),
            "train_pos_pct": float(np.mean(y_tr_ros) * 100),
            "test_pos_pct": float(np.mean(y_te_ros) * 100),
            "synthetic_samples_in_test": 0,
            "duplicate_samples_in_test": int(leaked_duplicate_count),
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "brier_score": m["brier_score"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "runtime_s": float(elapsed)
        })
    
    # -------------------------------------------------------------------------
    # PIPELINE 5: Scaling / Preprocessing Leakage Only (No Resampling)
    # Global preprocessor fit on all data -> split into train/test -> test on untouched prevalence
    # -------------------------------------------------------------------------
    log("\n[PIPELINE 5] Scaling/Preprocessing Leakage Only (Global Scaling, No Resampling)...")
    X_tr_scale = X_all[train_df.index]
    y_tr_scale = df["target"].iloc[train_df.index].values
    X_te_scale = X_all[test_df.index]
    y_te_scale = df["target"].iloc[test_df.index].values
    
    for model_name, model in [
        ("LogisticRegression", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)),
        ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=seed)),
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed))
    ]:
        t0 = time.time()
        model.fit(X_tr_scale, y_tr_scale)
        y_prob_scale = model.predict_proba(X_te_scale)[:, 1]
        m = compute_binary_metrics(y_te_scale, y_prob_scale, threshold=0.5)
        ci = compute_bootstrap_ci(y_te_scale, y_prob_scale, threshold=0.5, n_bootstraps=500, seed=seed)
        elapsed = time.time() - t0
        
        log(f"  {model_name}: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], Acc={m['accuracy']:.4f}, Recall={m['recall']:.4f}, F1={m['f1']:.4f}")
        results.append({
            "pipeline": "5_Leaked_Global_Scaling_Only",
            "model": model_name,
            "train_size": len(y_tr_scale),
            "test_size": len(y_te_scale),
            "train_pos_pct": float(np.mean(y_tr_scale) * 100),
            "test_pos_pct": float(np.mean(y_te_scale) * 100),
            "synthetic_samples_in_test": 0,
            "duplicate_samples_in_test": 0,
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "brier_score": m["brier_score"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
            "runtime_s": float(elapsed)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv("results/tables/leakage_comparison.csv", index=False)
    log("\nSaved results/tables/leakage_comparison.csv")
    
    # Save log file
    with open(log_file, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Saved {log_file}")
    
    # Generate visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: ROC-AUC across pipelines
    sns.barplot(data=res_df, x="pipeline", y="roc_auc", hue="model", ax=axes[0], palette="Set1")
    axes[0].set_title("Impact of Methodological Leakage on ROC-AUC", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("ROC-AUC", fontsize=12)
    axes[0].set_xlabel("Pipeline Condition", fontsize=12)
    axes[0].set_xticklabels(["Clean Baseline", "Clean SMOTE (Train)", "Leaked SMOTE", "Leaked RandomOverSampler", "Global Scaling Only"], rotation=30, ha="right")
    axes[0].set_ylim(0.75, 1.0)
    axes[0].axhline(0.8392, color="blue", linestyle=":", label="PDF Clean Baseline (0.8392)")
    axes[0].axhline(0.9243, color="red", linestyle="--", label="PDF Leaked Claim (0.9243)")
    axes[0].legend(loc="lower right")
    
    # Subplot 2: Accuracy vs Recall distortion
    sns.scatterplot(data=res_df, x="recall", y="accuracy", hue="pipeline", style="model", s=150, ax=axes[1])
    axes[1].set_title("Distortion in Accuracy vs Recall Across Pipelines", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig("results/figures/leakage_comparison.png", dpi=300)
    plt.close()
    log("Saved results/figures/leakage_comparison.png")
    
    return res_df

if __name__ == "__main__":
    run_leakage_audit()
