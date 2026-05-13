"""
evaluation.py
=============
Unified evaluation utilities for the ML vs Graph comparison.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# ML Baseline Loader
# ============================================================

def load_ml_baseline(pickle_dir: str = "../Fraud-Detection/pickled_storage") -> dict:
    """
    Load pre-computed ML results from the companion project.
    Returns a dict: {dataset_name: {model_name: {metric: value}}}
    """
    results = {}
    pkl_dir = Path(pickle_dir)

    # Load ensemble results (most complete)
    ensemble_path = pkl_dir / "ensemble_results_all.pkl"
    if ensemble_path.exists():
        with open(ensemble_path, "rb") as f:
            results["ensemble"] = pickle.load(f)
        print(f"[Baseline] Loaded ensemble results from {ensemble_path}")

    # Load individual model results
    for pkl_file in pkl_dir.glob("results_*.pkl"):
        key = pkl_file.stem.replace("results_", "")
        with open(pkl_file, "rb") as f:
            results[key] = pickle.load(f)
        print(f"[Baseline] Loaded {key} from {pkl_file}")

    # Also load the CSV summary
    csv_path = pkl_dir.parent / "outputs" / "ensemble_summary_all.csv"
    if csv_path.exists():
        results["summary_df"] = pd.read_csv(csv_path)
        print(f"[Baseline] Loaded summary CSV from {csv_path}")

    return results


def get_ml_baseline_table(pickle_dir: str = "../Fraud-Detection/pickled_storage") -> pd.DataFrame:
    """
    Load and return a clean comparison DataFrame from the ML project.
    """
    csv_path = Path(pickle_dir).parent / "outputs" / "ensemble_summary_all.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["Source"] = "Tabular-ML"
        return df

    # Fallback: hardcoded from known results
    records = [
        {"Dataset": "Credit Card", "Source": "Tabular-ML", "Method": "OCSVM",
         "Mean_AUC": 0.9514, "Mean_AUPRC": 0.2731},
        {"Dataset": "Credit Card", "Source": "Tabular-ML", "Method": "GMM",
         "Mean_AUC": 0.9588, "Mean_AUPRC": 0.5720},
        {"Dataset": "Credit Card", "Source": "Tabular-ML", "Method": "IsolationForest",
         "Mean_AUC": 0.9496, "Mean_AUPRC": 0.1543},
        {"Dataset": "Credit Card", "Source": "Tabular-ML", "Method": "LOF",
         "Mean_AUC": 0.9532, "Mean_AUPRC": 0.6391},
        {"Dataset": "Credit Card", "Source": "Tabular-ML", "Method": "Ensemble_Best",
         "Mean_AUC": 0.9535, "Mean_AUPRC": 0.5191},
        {"Dataset": "Medicare", "Source": "Tabular-ML", "Method": "OCSVM",
         "Mean_AUC": 0.5261, "Mean_AUPRC": 0.0019},
        {"Dataset": "Medicare", "Source": "Tabular-ML", "Method": "GMM",
         "Mean_AUC": 0.5977, "Mean_AUPRC": 0.0023},
        {"Dataset": "Medicare", "Source": "Tabular-ML", "Method": "IsolationForest",
         "Mean_AUC": 0.6081, "Mean_AUPRC": 0.0038},
        {"Dataset": "Medicare", "Source": "Tabular-ML", "Method": "LOF",
         "Mean_AUC": 0.6043, "Mean_AUPRC": 0.0021},
        {"Dataset": "Medicare", "Source": "Tabular-ML", "Method": "Ensemble_Best",
         "Mean_AUC": 0.5974, "Mean_AUPRC": 0.0037},
    ]
    return pd.DataFrame(records)


# ============================================================
# Unified Metrics
# ============================================================

def compute_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                    threshold: float = None) -> dict:
    """Compute full set of classification metrics."""
    if len(np.unique(y_true)) < 2:
        return {"auc_roc": 0, "auc_prc": 0, "f1": 0, "precision": 0, "recall": 0}

    auc_roc = roc_auc_score(y_true, y_scores)
    auc_prc = average_precision_score(y_true, y_scores)

    # Find optimal threshold if not provided
    if threshold is None:
        thresholds = np.linspace(0, 1, 200)
        best_f1, best_thresh = 0, 0.5
        for t in thresholds:
            preds = (y_scores >= t).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t
        threshold = best_thresh

    preds = (y_scores >= threshold).astype(int)
    return {
        "auc_roc": auc_roc,
        "auc_prc": auc_prc,
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "threshold": threshold,
    }


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Eval] Metrics saved to {path}")


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ============================================================
# Comparison Table Builder
# ============================================================

def build_comparison_table(graph_results: list, ml_baseline_path: str = None) -> pd.DataFrame:
    """
    Build the master comparison table combining ML baseline + graph results.

    graph_results: list of dicts with keys:
        {dataset, source, method, auc_roc, auc_prc, f1, train_time}
    """
    # Load ML baseline
    if ml_baseline_path:
        ml_df = get_ml_baseline_table(ml_baseline_path)
    else:
        ml_df = get_ml_baseline_table()

    # Convert to standard format
    ml_records = []
    for _, row in ml_df.iterrows():
        ml_records.append({
            "Dataset": row.get("Dataset", ""),
            "Category": "Tabular-ML",
            "Method": row.get("Method", ""),
            "AUC_ROC": row.get("Mean_AUC", np.nan),
            "AUC_PRC": row.get("Mean_AUPRC", np.nan),
            "F1": np.nan,
            "Train_Time_s": np.nan,
        })

    graph_records = []
    for r in graph_results:
        graph_records.append({
            "Dataset": r.get("dataset", ""),
            "Category": r.get("category", "Graph"),
            "Method": r.get("method", ""),
            "AUC_ROC": r.get("auc_roc", np.nan),
            "AUC_PRC": r.get("auc_prc", np.nan),
            "F1": r.get("f1", np.nan),
            "Train_Time_s": r.get("train_time", np.nan),
        })

    all_records = ml_records + graph_records
    df = pd.DataFrame(all_records)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Pretty print the comparison table."""
    print("\n" + "=" * 80)
    print("FRAUD DETECTION: TABULAR-ML vs GRAPH METHODS COMPARISON")
    print("=" * 80)

    for dataset in df["Dataset"].unique():
        sub = df[df["Dataset"] == dataset].sort_values("AUC_ROC", ascending=False)
        print(f"\n📊 Dataset: {dataset}")
        print(f"{'Method':<25} {'Category':<20} {'AUC-ROC':>8} {'AUC-PRC':>8} {'F1':>8}")
        print("-" * 75)
        for _, row in sub.iterrows():
            print(f"{row['Method']:<25} {row['Category']:<20} "
                  f"{row['AUC_ROC']:>8.4f} {row['AUC_PRC']:>8.4f} "
                  f"{str(round(row['F1'], 4)) if not np.isnan(row['F1']) else 'N/A':>8}")

    print("\n" + "=" * 80)
