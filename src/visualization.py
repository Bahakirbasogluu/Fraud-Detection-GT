"""
visualization.py
================
Graph visualization utilities for the Fraud Detection GT project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Style Setup
# ============================================================

FRAUD_COLOR = "#FF4444"
NORMAL_COLOR = "#4488FF"
COMMUNITY_PALETTE = "tab20"

def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "#0F0F1A",
        "axes.facecolor": "#1A1A2E",
        "axes.edgecolor": "#444466",
        "axes.labelcolor": "#CCCCEE",
        "xtick.color": "#CCCCEE",
        "ytick.color": "#CCCCEE",
        "text.color": "#EEEEFF",
        "grid.color": "#333355",
        "grid.alpha": 0.4,
        "font.family": "DejaVu Sans",
    })


# ============================================================
# Graph Structure Visualizations
# ============================================================

def plot_degree_distribution(G: nx.Graph, title: str = "Degree Distribution",
                              save_path: str = None):
    """Plot degree distribution with fraud vs normal overlay."""
    setup_style()
    degrees_fraud = [G.degree(n) for n in G.nodes() if G.nodes[n].get("label") == 1]
    degrees_normal = [G.degree(n) for n in G.nodes() if G.nodes[n].get("label") == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, color="#EEEEFF", fontweight="bold")

    # Log-scale degree distribution
    ax = axes[0]
    all_degrees = [G.degree(n) for n in G.nodes()]
    degree_counts = pd.Series(all_degrees).value_counts().sort_index()
    ax.loglog(degree_counts.index, degree_counts.values, "o-", color="#7777FF",
              alpha=0.8, markersize=4, label="All nodes")
    ax.set_xlabel("Degree (log scale)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Degree Distribution (Log-Log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fraud vs Normal degree comparison
    ax = axes[1]
    ax.hist(degrees_normal, bins=50, alpha=0.6, color=NORMAL_COLOR,
            label=f"Normal (n={len(degrees_normal)})", density=True)
    ax.hist(degrees_fraud, bins=50, alpha=0.6, color=FRAUD_COLOR,
            label=f"Fraud (n={len(degrees_fraud)})", density=True)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Density")
    ax.set_title("Degree: Fraud vs Normal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


def plot_centrality_comparison(df: pd.DataFrame, title: str = "Centrality: Fraud vs Normal",
                                save_path: str = None):
    """Boxplot comparison of centrality metrics for fraud vs normal nodes."""
    setup_style()
    metrics = ["pagerank", "betweenness_centrality", "closeness_centrality",
               "eigenvector_centrality", "clustering_coeff"]
    metrics = [m for m in metrics if m in df.columns]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, color="#EEEEFF", fontweight="bold", y=1.02)

    df_plot = df[df["label"].isin([0, 1])].copy()
    df_plot["label_str"] = df_plot["label"].map({0: "Normal", 1: "Fraud"})

    for ax, metric in zip(axes, metrics):
        data_normal = df_plot[df_plot["label"] == 0][metric].values
        data_fraud = df_plot[df_plot["label"] == 1][metric].values

        bp = ax.boxplot([data_normal, data_fraud], labels=["Normal", "Fraud"],
                         patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(NORMAL_COLOR + "88")
        bp["boxes"][1].set_facecolor(FRAUD_COLOR + "88")
        for median in bp["medians"]:
            median.set_color("#FFFFFF")

        ax.set_title(metric.replace("_", "\n"), fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


def plot_community_fraud_heatmap(community_df: pd.DataFrame, title: str = "Community Fraud Rates",
                                  save_path: str = None, top_n: int = 30):
    """Bar chart of fraud rates per community (top N by fraud rate)."""
    setup_style()
    df = community_df.nlargest(top_n, "fraud_rate")

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = [FRAUD_COLOR if r > 0.1 else NORMAL_COLOR for r in df["fraud_rate"]]
    bars = ax.bar(range(len(df)), df["fraud_rate"], color=colors, alpha=0.8, edgecolor="#333")
    ax.set_xlabel("Community (sorted by fraud rate)")
    ax.set_ylabel("Fraud Rate")
    ax.set_title(title, color="#EEEEFF")
    ax.axhline(df["fraud_rate"].mean(), color="#FFD700", linestyle="--",
               alpha=0.7, label=f"Mean: {df['fraud_rate'].mean():.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spectral_eigenvalues(eigenvalues: np.ndarray, title: str = "Laplacian Eigenvalue Spectrum",
                               save_path: str = None):
    """Plot Laplacian eigenvalues (spectral gap visualization)."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, color="#EEEEFF", fontweight="bold")

    ax = axes[0]
    ax.plot(eigenvalues, "o-", color="#7788FF", markersize=5, alpha=0.8)
    ax.axvline(1, color=FRAUD_COLOR, linestyle="--", alpha=0.7, label="Fiedler value (λ₂)")
    ax.set_xlabel("Index k")
    ax.set_ylabel("Eigenvalue λₖ")
    ax.set_title("Eigenvalue Spectrum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    gaps = np.diff(eigenvalues)
    ax.bar(range(len(gaps)), gaps, color="#55AAFF", alpha=0.8, edgecolor="#333")
    ax.set_xlabel("k")
    ax.set_ylabel("λₖ₊₁ − λₖ")
    ax.set_title("Spectral Gaps")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# Comparison Visualizations
# ============================================================

def plot_method_comparison(df: pd.DataFrame, dataset_name: str,
                            metric: str = "AUC_ROC", save_path: str = None):
    """
    Horizontal bar chart comparing all methods on a given dataset.
    Groups: Tabular-ML vs Graph-Classical vs Graph-Spectral vs Graph-GNN.
    """
    setup_style()
    sub = df[df["Dataset"] == dataset_name].dropna(subset=[metric]).sort_values(metric)

    category_colors = {
        "Tabular-ML": "#5577FF",
        "Graph-Classical": "#55DDAA",
        "Graph-Spectral": "#FFAA33",
        "Graph-GNN": "#FF5577",
        "Graph-Unsup": "#AA55FF",
    }

    colors = [category_colors.get(cat, "#888888") for cat in sub["Category"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(sub) * 0.4)))
    bars = ax.barh(sub["Method"], sub[metric], color=colors, alpha=0.85, edgecolor="#333")

    ax.set_xlabel(metric.replace("_", " "), fontsize=12)
    ax.set_title(f"{dataset_name}: {metric} Comparison", color="#EEEEFF", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.grid(True, axis="x", alpha=0.3)

    # Value labels
    for bar, val in zip(bars, sub[metric]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#EEEEFF")

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8)
               for c in category_colors.values()]
    ax.legend(handles, category_colors.keys(), loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Saved: {save_path}")
    plt.show()


def plot_roc_curves(roc_results: dict, title: str = "ROC Curves", save_path: str = None):
    """Plot multiple ROC curves on the same figure."""
    from sklearn.metrics import roc_curve, auc

    setup_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_results)))

    for (name, (y_true, y_score)), color in zip(roc_results.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, color="#EEEEFF")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
