"""
Experiment 5: Cross-Group Transfer & Subgroup Generalization Analysis
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
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_clean_data, build_preprocessor, load_config
from src.metrics import compute_binary_metrics, compute_bootstrap_ci

def run_cross_group_transfer():
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 5: CROSS-GROUP GENERALIZATION & TRANSFER MATRICES", flush=True)
    print("=" * 70, flush=True)
    
    df, train_df, test_df, config = load_clean_data()
    seed = config["project"]["random_seed"]
    all_features = config["feature_groups"]["group_d_all_features"]
    
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # PART A: AGE GROUP TRANSFER
    # Categories:
    # Young (<50): 18-24, 25-29, 30-34, 35-39, 40-44, 45-49
    # Middle (50-69): 50-54, 55-59, 60-64, 65-69
    # Elderly (70+): 70-74, 75-79, 80 or older
    # -------------------------------------------------------------------------
    print("\n--- Running Age Group Transfer Matrix ---", flush=True)
    young_cats = config["subgroups"]["age"]["young"]
    middle_cats = config["subgroups"]["age"]["middle"]
    elderly_cats = config["subgroups"]["age"]["elderly"]
    
    def assign_age_group(age_cat):
        if age_cat in young_cats:
            return "Young_<50"
        elif age_cat in middle_cats:
            return "Middle_50-69"
        elif age_cat in elderly_cats:
            return "Elderly_70+"
        return "Unknown"
    
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["AgeGroup"] = train_df["AgeCategory"].apply(assign_age_group)
    test_df["AgeGroup"] = test_df["AgeCategory"].apply(assign_age_group)
    
    age_groups = ["Young_<50", "Middle_50-69", "Elderly_70+"]
    
    # When training on a specific age group, we exclude AgeCategory feature to evaluate pure transfer of clinical/lifestyle risk factors
    age_transfer_features = [f for f in all_features if f != "AgeCategory"]
    num_cols = [c for c in config["features"]["numerical"] if c in age_transfer_features]
    cat_cols = [c for c in config["features"]["categorical"] if c in age_transfer_features]
    
    # Train separate models per age group
    age_models = {}
    for grp in age_groups:
        sub_train = train_df[train_df["AgeGroup"] == grp]
        pre = build_preprocessor(num_cols, cat_cols)
        X_sub_tr = pre.fit_transform(sub_train[age_transfer_features])
        y_sub_tr = sub_train["target"].values
        
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
        clf.fit(X_sub_tr, y_sub_tr)
        age_models[grp] = (pre, clf, len(y_sub_tr), float(np.mean(y_sub_tr)))
        print(f"  Trained model on {grp} (N={len(y_sub_tr)}, Pos %={np.mean(y_sub_tr)*100:.2f}%)", flush=True)
        
    # Also train a pooled model on all training data
    pre_pooled = build_preprocessor(num_cols, cat_cols)
    X_pool_tr = pre_pooled.fit_transform(train_df[age_transfer_features])
    y_pool_tr = train_df["target"].values
    clf_pooled = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    clf_pooled.fit(X_pool_tr, y_pool_tr)
    
    # Evaluate transfer matrix across all age groups
    age_transfer_records = []
    auc_matrix = pd.DataFrame(index=age_groups + ["Pooled_All"], columns=age_groups)
    
    for src_name in age_groups + ["Pooled_All"]:
        if src_name == "Pooled_All":
            src_pre, src_clf = pre_pooled, clf_pooled
            src_n, src_prev = len(train_df), float(np.mean(train_df["target"]))
        else:
            src_pre, src_clf, src_n, src_prev = age_models[src_name]
            
        for tgt_name in age_groups:
            tgt_test = test_df[test_df["AgeGroup"] == tgt_name]
            X_tgt_te = src_pre.transform(tgt_test[age_transfer_features])
            y_tgt_te = tgt_test["target"].values
            
            y_prob = src_clf.predict_proba(X_tgt_te)[:, 1]
            m = compute_binary_metrics(y_tgt_te, y_prob, threshold=0.5)
            ci = compute_bootstrap_ci(y_tgt_te, y_prob, threshold=0.5, n_bootstraps=500, seed=seed)
            
            auc_matrix.loc[src_name, tgt_name] = m["roc_auc"]
            
            is_in_group = (src_name == tgt_name)
            age_transfer_records.append({
                "domain": "Age",
                "source_group": src_name,
                "target_group": tgt_name,
                "is_in_group": is_in_group,
                "source_train_n": src_n,
                "source_train_prevalence": src_prev,
                "target_test_n": len(y_tgt_te),
                "target_test_prevalence": float(np.mean(y_tgt_te)),
                "roc_auc": m["roc_auc"],
                "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
                "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
                "pr_auc": m["pr_auc"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "mcc": m["mcc"]
            })
            print(f"  Transfer [{src_name} -> {tgt_name}]: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], PR-AUC={m['pr_auc']:.4f}, Tgt Prev={np.mean(y_tgt_te)*100:.2f}%", flush=True)
            
    age_transfer_df = pd.DataFrame(age_transfer_records)
    age_transfer_df.to_csv("results/tables/age_transfer_matrix.csv", index=False)
    
    # -------------------------------------------------------------------------
    # PART B: SEX GROUP TRANSFER (Male <-> Female)
    # -------------------------------------------------------------------------
    print("\n--- Running Sex Group Transfer Matrix ---", flush=True)
    sex_groups = ["Male", "Female"]
    sex_transfer_features = [f for f in all_features if f != "Sex"]
    num_cols_sex = [c for c in config["features"]["numerical"] if c in sex_transfer_features]
    cat_cols_sex = [c for c in config["features"]["categorical"] if c in sex_transfer_features]
    
    sex_models = {}
    for s in sex_groups:
        sub_train = train_df[train_df["Sex"] == s]
        pre = build_preprocessor(num_cols_sex, cat_cols_sex)
        X_sub_tr = pre.fit_transform(sub_train[sex_transfer_features])
        y_sub_tr = sub_train["target"].values
        
        clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
        clf.fit(X_sub_tr, y_sub_tr)
        sex_models[s] = (pre, clf, len(y_sub_tr), float(np.mean(y_sub_tr)))
        print(f"  Trained model on {s} (N={len(y_sub_tr)}, Pos %={np.mean(y_sub_tr)*100:.2f}%)", flush=True)
        
    pre_sex_pooled = build_preprocessor(num_cols_sex, cat_cols_sex)
    X_pool_sex_tr = pre_sex_pooled.fit_transform(train_df[sex_transfer_features])
    clf_sex_pooled = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    clf_sex_pooled.fit(X_pool_sex_tr, y_pool_tr)
    
    sex_transfer_records = []
    for src_name in sex_groups + ["Pooled_All"]:
        if src_name == "Pooled_All":
            src_pre, src_clf = pre_sex_pooled, clf_sex_pooled
            src_n, src_prev = len(train_df), float(np.mean(train_df["target"]))
        else:
            src_pre, src_clf, src_n, src_prev = sex_models[src_name]
            
        for tgt_name in sex_groups:
            tgt_test = test_df[test_df["Sex"] == tgt_name]
            X_tgt_te = src_pre.transform(tgt_test[sex_transfer_features])
            y_tgt_te = tgt_test["target"].values
            
            y_prob = src_clf.predict_proba(X_tgt_te)[:, 1]
            m = compute_binary_metrics(y_tgt_te, y_prob, threshold=0.5)
            ci = compute_bootstrap_ci(y_tgt_te, y_prob, threshold=0.5, n_bootstraps=500, seed=seed)
            
            sex_transfer_records.append({
                "domain": "Sex",
                "source_group": src_name,
                "target_group": tgt_name,
                "is_in_group": (src_name == tgt_name),
                "source_train_n": src_n,
                "source_train_prevalence": src_prev,
                "target_test_n": len(y_tgt_te),
                "target_test_prevalence": float(np.mean(y_tgt_te)),
                "roc_auc": m["roc_auc"],
                "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
                "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
                "pr_auc": m["pr_auc"],
                "recall": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "balanced_accuracy": m["balanced_accuracy"],
                "mcc": m["mcc"]
            })
            print(f"  Transfer [{src_name} -> {tgt_name}]: ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], PR-AUC={m['pr_auc']:.4f}, Tgt Prev={np.mean(y_tgt_te)*100:.2f}%", flush=True)
            
    sex_transfer_df = pd.DataFrame(sex_transfer_records)
    sex_transfer_df.to_csv("results/tables/sex_transfer_matrix.csv", index=False)
    
    # -------------------------------------------------------------------------
    # PART C: RACE SUBGROUP ANALYSIS
    # -------------------------------------------------------------------------
    print("\n--- Running Race Subgroup Disparities Analysis ---", flush=True)
    races = config["subgroups"]["race"]
    race_records = []
    
    # Evaluate the full pooled model across all racial subgroups on test data
    pre_full = build_preprocessor(
        [c for c in config["features"]["numerical"] if c in all_features],
        [c for c in config["features"]["categorical"] if c in all_features]
    )
    X_full_tr = pre_full.fit_transform(train_df[all_features])
    clf_full = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss", random_state=seed)
    clf_full.fit(X_full_tr, train_df["target"].values)
    
    for r in races:
        sub_test = test_df[test_df["Race"] == r]
        if len(sub_test) < 50:
            continue
        X_sub_te = pre_full.transform(sub_test[all_features])
        y_sub_te = sub_test["target"].values
        y_prob = clf_full.predict_proba(X_sub_te)[:, 1]
        
        m = compute_binary_metrics(y_sub_te, y_prob, threshold=0.5)
        ci = compute_bootstrap_ci(y_sub_te, y_prob, threshold=0.5, n_bootstraps=500, seed=seed)
        
        race_records.append({
            "race_group": r,
            "test_sample_size": len(sub_test),
            "percentage_of_test": float(len(sub_test) / len(test_df) * 100),
            "prevalence_pct": float(np.mean(y_sub_te) * 100),
            "roc_auc": m["roc_auc"],
            "roc_auc_ci_lower": ci["roc_auc"]["ci_lower"],
            "roc_auc_ci_upper": ci["roc_auc"]["ci_upper"],
            "pr_auc": m["pr_auc"],
            "brier_score": m["brier_score"],
            "recall": m["recall"],
            "specificity": m["specificity"],
            "f1": m["f1"]
        })
        print(f"  Race [{r}] N={len(sub_test)} ({len(sub_test)/len(test_df)*100:.1f}%), Prev={np.mean(y_sub_te)*100:.2f}% | ROC-AUC={m['roc_auc']:.4f} [{ci['roc_auc']['ci_lower']:.4f}, {ci['roc_auc']['ci_upper']:.4f}], PR-AUC={m['pr_auc']:.4f}", flush=True)
        
    race_df = pd.DataFrame(race_records)
    race_df.to_csv("results/tables/race_transfer_matrix.csv", index=False)
    
    # Combined subgroup performance table
    combined_subgroups = pd.concat([
        age_transfer_df[age_transfer_df["source_group"] == "Pooled_All"].assign(subgroup_type="Age"),
        sex_transfer_df[sex_transfer_df["source_group"] == "Pooled_All"].assign(subgroup_type="Sex"),
        race_df.rename(columns={"race_group": "target_group"}).assign(subgroup_type="Race", source_group="Pooled_All")
    ], ignore_index=True)
    combined_subgroups.to_csv("results/tables/subgroup_performance.csv", index=False)
    
    # -------------------------------------------------------------------------
    # PART D: FIGURES (Heatmaps and Subgroup Performance)
    # -------------------------------------------------------------------------
    print("\n--- Generating Transfer Heatmap and Figures ---", flush=True)
    plt.figure(figsize=(7, 6))
    auc_num = auc_matrix.astype(float)
    sns.heatmap(auc_num, annot=True, fmt=".4f", cmap="YlGnBu", vmin=0.68, vmax=0.85, cbar_kws={'label': 'Test ROC-AUC'})
    plt.title("Age-Stratified Cross-Group Generalization Matrix\n(Source Model vs Target Population)", fontsize=13, fontweight="bold")
    plt.xlabel("Target Evaluation Cohort", fontsize=11)
    plt.ylabel("Source Training Cohort", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/figures/age_transfer_heatmap.png", dpi=300)
    plt.close()
    
    # Figure: Subgroup Performance by Race and Age
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Subplot 1: Subgroup ROC-AUC by Race
    sns.barplot(data=race_df, x="race_group", y="roc_auc", palette="Blues_d", ax=axes[0])
    axes[0].set_title("Pooled Model ROC-AUC across Racial Subgroups", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Test ROC-AUC", fontsize=11)
    axes[0].set_xlabel("Racial Subgroup", fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].set_ylim(0.70, 0.88)
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
                         
    # Subplot 2: Prevalence vs PR-AUC Disparity
    sns.scatterplot(data=race_df, x="prevalence_pct", y="pr_auc", s=200, color="darkred", ax=axes[1])
    for _, row in race_df.iterrows():
        axes[1].text(row["prevalence_pct"] + 0.15, row["pr_auc"], row["race_group"], fontsize=10)
    axes[1].set_title("Prevalence Effect on PR-AUC across Racial Subgroups", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Subgroup Disease Prevalence (%)", fontsize=11)
    axes[1].set_ylabel("Subgroup Average Precision (PR-AUC)", fontsize=11)
    
    plt.tight_layout()
    plt.savefig("results/figures/subgroup_performance.png", dpi=300)
    plt.close()
    
    print("Saved all tables and figures for Experiment 5.", flush=True)

if __name__ == "__main__":
    run_cross_group_transfer()
