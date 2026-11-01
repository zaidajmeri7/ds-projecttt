"""
Experiment 6: Model Interpretation, Permutation Importance & SHAP
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
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import shap

from src.data_loader import load_clean_data, build_preprocessor, load_config

def run_interpretability():
    print("=" * 70, flush=True)
    print("STARTING MODEL INTERPRETATION & EXPLAINABILITY ANALYSIS", flush=True)
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
    
    # Preprocessor strictly on train
    pre = build_preprocessor(num_cols, cat_cols)
    X_tr = pre.fit_transform(train_df[all_features])
    X_te = pre.transform(test_df[all_features])
    
    # Get feature names from preprocessor
    feature_names = []
    if num_cols:
        feature_names.extend(num_cols)
    if cat_cols:
        cat_encoder = pre.named_transformers_["cat"]
        encoded_cats = cat_encoder.get_feature_names_out(cat_cols)
        feature_names.extend(encoded_cats)
        
    print(f"Total encoded features: {len(feature_names)}", flush=True)
    
    # Train full XGBoost model
    print("Training full XGBoost model for interpretation...", flush=True)
    clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    clf.fit(X_tr, y_train)
    
    # -------------------------------------------------------------------------
    # 1. Permutation Importance on Test Set
    # -------------------------------------------------------------------------
    print("Calculating permutation importance on test set (5 repeats, scoring=roc_auc)...", flush=True)
    t0 = time.time()
    perm_res = permutation_importance(
        clf, X_te, y_test, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1
    )
    print(f"Permutation importance completed in {time.time()-t0:.1f}s", flush=True)
    
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": perm_res.importances_mean,
        "importance_std": perm_res.importances_std
    }).sort_values(by="importance_mean", ascending=False)
    
    # Classify feature origin
    def get_feature_category(feat):
        base = feat.split("_")[0]
        if base in config["feature_groups"]["group_a_demographics"]:
            return "Demographic"
        elif base in config["feature_groups"]["group_b_lifestyle"]:
            return "Lifestyle"
        elif base in config["feature_groups"]["group_c_health_status"]:
            return "Health Status"
        return "Other"
        
    perm_df["category"] = perm_df["feature"].apply(get_feature_category)
    perm_df.to_csv("results/tables/feature_importance.csv", index=False)
    print("Saved results/tables/feature_importance.csv", flush=True)
    
    # -------------------------------------------------------------------------
    # 2. Feature Stability Across 5 Cross-Validation Folds
    # -------------------------------------------------------------------------
    print("\nAssessing feature importance stability across 5 CV folds...", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_importances = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, y_train)):
        f_tr, f_val = train_df.iloc[tr_idx], train_df.iloc[val_idx]
        fold_pre = build_preprocessor(num_cols, cat_cols)
        X_f_tr = fold_pre.fit_transform(f_tr[all_features])
        X_f_val = fold_pre.transform(f_val[all_features])
        
        fold_clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
        fold_clf.fit(X_f_tr, f_tr["target"].values)
        
        # Gain-based importance from XGBoost trees
        fold_gains = fold_clf.feature_importances_
        fold_importances.append(fold_gains)
        
    fold_imp_matrix = np.array(fold_importances) # shape: (5, n_features)
    stability_df = pd.DataFrame({
        "feature": feature_names,
        "mean_gain": np.mean(fold_imp_matrix, axis=0),
        "std_gain": np.std(fold_imp_matrix, axis=0),
        "cv_rank_stability": np.std(fold_imp_matrix, axis=0) / (np.mean(fold_imp_matrix, axis=0) + 1e-8)
    }).sort_values(by="mean_gain", ascending=False)
    stability_df["category"] = stability_df["feature"].apply(get_feature_category)
    stability_df.to_csv("results/tables/feature_stability.csv", index=False)
    print("Saved results/tables/feature_stability.csv", flush=True)
    
    # -------------------------------------------------------------------------
    # 3. TreeSHAP Explanation on Representative Test Sample
    # -------------------------------------------------------------------------
    print("\nGenerating SHAP explanations using TreeExplainer...", flush=True)
    # Stratified subsample of test set (N=2,000) for fast, robust SHAP calculation
    sample_idx = np.random.RandomState(seed).choice(len(y_test), size=2000, replace=False)
    X_sample = X_te[sample_idx]
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=15)
    plt.title("TreeSHAP Summary Plot (Top 15 Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/shap_summary.png", dpi=300)
    plt.close()
    print("Saved results/figures/shap_summary.png", flush=True)
    
    # -------------------------------------------------------------------------
    # 4. Feature Importance Comparison Figure (Top 15 by Permutation Importance)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 7))
    top15 = perm_df.head(15)
    sns.barplot(data=top15, x="importance_mean", y="feature", hue="category", dodge=False, palette="Set2")
    plt.title("Top 15 Features by Test Set Permutation Importance (ROC-AUC Loss)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean ROC-AUC Drop When Permuted", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/figures/feature_importance_comparison.png", dpi=300)
    plt.close()
    print("Saved results/figures/feature_importance_comparison.png", flush=True)

if __name__ == "__main__":
    run_interpretability()
