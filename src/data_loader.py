"""
Data Loading, Verification, and Preprocessing Utilities
Project: Heart Disease Prediction and Risk-Factor Analysis
"""

import os
import hashlib
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_config(config_path="configs/experiment_config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def verify_dataset(filepath, expected_sha256):
    """Verifies that the dataset exists and matches the expected SHA-256 hash."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    
    if file_hash != expected_sha256:
        raise ValueError(
            f"Dataset hash mismatch!\nExpected: {expected_sha256}\nFound:    {file_hash}"
        )
    return file_hash

def get_dataset_summary(df, target_col="HeartDisease"):
    """Returns key dataset descriptive statistics."""
    summary = {
        "total_records": int(len(df)),
        "total_columns": int(df.shape[1]),
        "target_column": target_col,
        "class_counts": {k: int(v) for k, v in df[target_col].value_counts().to_dict().items()},
        "class_proportions": {k: float(v) for k, v in df[target_col].value_counts(normalize=True).to_dict().items()},
        "positive_class_percentage": float((df[target_col] == "Yes").mean() * 100),
        "missing_values_total": int(df.isnull().sum().sum()),
        "features": [c for c in df.columns if c != target_col]
    }
    return summary

def load_clean_data(config_path="configs/experiment_config.yaml"):
    """
    Loads dataset, verifies integrity, encodes target as binary (0/1),
    and creates stratified train/test split.
    """
    config = load_config(config_path)
    data_path = config["data"]["raw_path"]
    expected_hash = config["data"]["dataset_sha256"]
    
    verify_dataset(data_path, expected_hash)
    df = pd.read_csv(data_path)
    
    target_col = config["data"]["target_column"]
    # Encode target: Yes -> 1, No -> 0
    df["target"] = (df[target_col] == config["data"]["positive_label"]).astype(int)
    
    # Stratified Train/Test Split
    test_size = config["data"]["test_size"]
    seed = config["project"]["random_seed"]
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=seed
    )
    
    return df, train_df, test_df, config

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Builds a scikit-learn ColumnTransformer that scales numerical columns
    and one-hot encodes categorical columns with drop='first' (or handle_unknown='ignore').
    """
    transformers = []
    if numerical_cols:
        transformers.append(
            ("num", StandardScaler(), numerical_cols)
        )
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols)
        )
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    return preprocessor

if __name__ == "__main__":
    df, train_df, test_df, config = load_clean_data()
    summary = get_dataset_summary(df)
    print("Dataset verification passed!")
    print(f"Total rows: {summary['total_records']}, Target %: {summary['positive_class_percentage']:.2f}%")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
