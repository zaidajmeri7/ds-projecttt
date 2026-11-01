"""
Evaluation Metrics, Calibration, and Statistical Uncertainty Utilities
Project: Heart Disease Prediction and Risk-Factor Analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    precision_recall_curve
)

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes discrimination, calibration, and threshold-dependent classification metrics.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-15, 1.0 - 1e-15))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "prevalence": float(np.mean(y_true))
    }
    return metrics

def compute_ece(y_true, y_prob, n_bins=10, strategy="uniform"):
    """
    Computes Expected Calibration Error (ECE).
    strategy: 'uniform' (equal width bins) or 'quantile' (equal sample bins).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    
    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ece = 0.0
    total_samples = len(y_true)
    bin_stats = []
    
    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            abs_diff = np.abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * abs_diff
            bin_stats.append({
                "bin": b,
                "count": int(bin_count),
                "accuracy": float(bin_acc),
                "confidence": float(bin_conf),
                "abs_error": float(abs_diff)
            })
    return float(ece), bin_stats

def fast_binary_auc(y_true, y_prob):
    """Fast rank-sum ROC-AUC calculation (exact match to sklearn.metrics.roc_auc_score)."""
    order = np.argsort(y_prob)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(y_prob)) + 1
    pos_mask = (y_true == 1)
    n_pos = np.sum(pos_mask)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((np.sum(rank[pos_mask]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def compute_bootstrap_ci(y_true, y_prob, threshold=0.5, n_bootstraps=200, alpha=0.05, seed=42):
    """
    Computes 95% bootstrap percentile confidence intervals for key metrics on test data.
    Highly optimized vector operations for large medical tabular datasets.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    n_samples = len(y_true)
    
    boot_roc_auc = []
    boot_pr_auc = []
    boot_brier = []
    boot_f1 = []
    boot_recall = []
    boot_precision = []
    boot_specificity = []
    boot_bal_acc = []
    boot_mcc = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        st = y_true[idx]
        sp = y_prob[idx]
        
        n_pos = np.count_nonzero(st == 1)
        if n_pos == 0 or n_pos == n_samples:
            continue
            
        pred = (sp >= threshold)
        tp = np.count_nonzero(pred & (st == 1))
        fp = np.count_nonzero(pred & (st == 0))
        tn = np.count_nonzero((~pred) & (st == 0))
        fn = np.count_nonzero((~pred) & (st == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        bal_acc = 0.5 * (rec + spec)
        
        denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
        mcc = float(tp * tn - fp * fn) / denom if denom > 0 else 0.0
        
        brier = float(np.mean((sp - st) ** 2))
        auc = fast_binary_auc(st, sp)
        # Periodic PR-AUC check (or sklearn)
        pr_auc = float(average_precision_score(st, sp))
        
        boot_roc_auc.append(auc)
        boot_pr_auc.append(pr_auc)
        boot_brier.append(brier)
        boot_f1.append(f1)
        boot_recall.append(rec)
        boot_precision.append(prec)
        boot_specificity.append(spec)
        boot_bal_acc.append(bal_acc)
        boot_mcc.append(mcc)
        
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)
    
    ci_results = {}
    for k, vals in [
        ("roc_auc", boot_roc_auc),
        ("pr_auc", boot_pr_auc),
        ("brier_score", boot_brier),
        ("f1", boot_f1),
        ("recall", boot_recall),
        ("precision", boot_precision),
        ("specificity", boot_specificity),
        ("balanced_accuracy", boot_bal_acc),
        ("mcc", boot_mcc)
    ]:
        if len(vals) > 0:
            ci_results[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_lower": float(np.percentile(vals, lower_pct)),
                "ci_upper": float(np.percentile(vals, upper_pct))
            }
    return ci_results
