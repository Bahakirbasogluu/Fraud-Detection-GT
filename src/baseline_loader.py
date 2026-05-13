"""
baseline_loader.py
==================
Loads pickled model results from the companion ML project
(Fraud-Detection / BLG607 project) for comparison.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_results(pickle_dir: str = "../Fraud-Detection/pickled_storage") -> dict:
    """Load all pickled results from the ML project."""
    pkl_dir = Path(pickle_dir)
    results = {}

    for pkl_file in sorted(pkl_dir.glob("*.pkl")):
        try:
            with open(pkl_file, "rb") as f:
                results[pkl_file.stem] = pickle.load(f)
            print(f"[Baseline] Loaded: {pkl_file.name}")
        except Exception as e:
            print(f"[Baseline] Failed to load {pkl_file.name}: {e}")

    return results


def get_summary_csv(pickle_dir: str = "../Fraud-Detection/pickled_storage") -> pd.DataFrame:
    """Load the pre-computed summary CSV if available."""
    csv_path = Path(pickle_dir).parent / "outputs" / "ensemble_summary_all.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"[Baseline] Loaded summary CSV: {csv_path}")
        return df
    else:
        print(f"[Baseline] Summary CSV not found at {csv_path}")
        return None


# Hardcoded baseline for when files are not accessible
HARDCODED_BASELINE = {
    "Credit Card": {
        "OCSVM": {"AUC_ROC": 0.9514, "AUC_PRC": 0.2731},
        "GMM": {"AUC_ROC": 0.9588, "AUC_PRC": 0.5720},
        "IsolationForest": {"AUC_ROC": 0.9496, "AUC_PRC": 0.1543},
        "LOF": {"AUC_ROC": 0.9532, "AUC_PRC": 0.6391},
        "Soft_Voting": {"AUC_ROC": 0.9535, "AUC_PRC": 0.4672},
        "Weighted_Voting": {"AUC_ROC": 0.9535, "AUC_PRC": 0.4675},
        "Stacking": {"AUC_ROC": 0.9532, "AUC_PRC": 0.5191},
    },
    "Medicare": {
        "OCSVM": {"AUC_ROC": 0.5261, "AUC_PRC": 0.0019},
        "GMM": {"AUC_ROC": 0.5977, "AUC_PRC": 0.0023},
        "IsolationForest": {"AUC_ROC": 0.6081, "AUC_PRC": 0.0038},
        "LOF": {"AUC_ROC": 0.6043, "AUC_PRC": 0.0021},
        "Soft_Voting": {"AUC_ROC": 0.5840, "AUC_PRC": 0.0031},
        "Weighted_Voting": {"AUC_ROC": 0.5877, "AUC_PRC": 0.0032},
        "Stacking": {"AUC_ROC": 0.5974, "AUC_PRC": 0.0037},
    },
}
