# Fraud Detection via Graph Theory
## Graph Theory Term Project — Graduate Level

> **Comparison Study**: Tabular ML (One-Class Classification) vs. Graph-Based Methods (Classical Graph Analysis + GNN)
> Uses the **same datasets** as the companion ML project to enable direct performance comparison.

---

## Research Hypothesis

> *"Fraudulent entities form structurally distinct communities in transaction graphs, and graph-theoretic methods (GNN, spectral analysis, centrality) significantly outperform tabular anomaly detection on relational fraud data — especially for Medicare provider-level fraud."*

---

## Datasets

| Dataset | Type | Source | ML Baseline AUC |
|---------|------|--------|-----------------|
| Credit Card Fraud | Tabular → Similarity Graph | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | 0.959 (GMM) |
| Medicare Fraud | Tabular → Provider-Drug Graph | [CMS](https://data.cms.gov/) + [LEIE](https://oig.hhs.gov/) | 0.608 (IsoForest) |

---

## Graph Methods

### Classical Graph Analysis
- PageRank, Betweenness, Closeness, Eigenvector Centrality
- Community Detection: Louvain, Girvan-Newman
- Clustering Coefficient, Degree Distribution
- Motif Analysis (triangles, star patterns)

### Spectral Methods
- Graph Laplacian eigendecomposition
- Fiedler vector & algebraic connectivity
- Spectral Clustering
- Chebyshev polynomial approximation

### Graph Neural Networks (PyTorch Geometric)
- **GCN** — Graph Convolutional Network (Kipf & Welling, 2017)
- **GraphSAGE** — Inductive representation learning (Hamilton et al., 2017)
- **GAT** — Graph Attention Network (Veličković et al., 2018)
- **DOMINANT** — Deep Anomaly Detection on Attributed Networks (Ding et al., 2019)

---

## Project Structure

```
Fraud-Detection-GT/
├── data/
│   ├── raw/                    # Original datasets (gitignored)
│   ├── processed/              # Graph-ready processed data
│   └── graphs/                 # Saved graph objects
├── notebooks/                  # Generated Jupyter notebooks
│   ├── 01_EDA_and_Baseline.ipynb
│   ├── 02_Graph_Construction.ipynb
│   ├── 03_Classical_Graph_Analysis.ipynb
│   ├── 04_Spectral_Methods.ipynb
│   ├── 05_GNN_Models.ipynb
│   └── 06_Comparison_and_Results.ipynb
├── scripts/                    # Notebook generator scripts (.py → .ipynb)
│   ├── generate_notebooks.py   # Run this to create all notebooks
│   ├── nb_01_eda_baseline.py
│   ├── nb_02_graph_construction.py
│   ├── nb_03_classical_graph_analysis.py
│   ├── nb_04_spectral_methods.py
│   ├── nb_05_gnn_models.py
│   └── nb_06_comparison_results.py
├── src/                        # Reusable Python modules
│   ├── graph_builder.py        # Graph construction utilities
│   ├── graph_features.py       # Centrality, community, spectral features
│   ├── gnn_models.py           # GCN, GraphSAGE, GAT, DOMINANT
│   ├── baseline_loader.py      # Load ML results from sibling project
│   ├── evaluation.py           # Unified metrics & comparison
│   └── visualization.py        # Graph visualization utilities
├── outputs/
│   ├── figures/                # All plots
│   ├── metrics/                # JSON/CSV results
│   ├── models/                 # Saved GNN weights
│   └── graphs/                 # Graph visualizations
├── reports/
│   ├── methodology.md
│   ├── results_analysis.md
│   └── comparison_ml_vs_graph.md
├── tests/
│   ├── test_graph_builder.py
│   └── test_evaluation.py
├── config.yaml                 # Centralized configuration
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Datasets
```
data/raw/creditcard.csv
data/raw/medicare/Combined_LEIE_Medicare_2017_2019_DOWNSIZED_1mil.csv
```

### 3. Generate All Notebooks
```bash
python scripts/generate_notebooks.py
```

### 4. Run Notebooks in Order
```
01 → 02 → 03 → 04 → 05 → 06
```

---

## ML Baseline (from companion project)

Loaded from: `../Fraud-Detection/pickled_storage/`

| Model | Credit Card AUC | Medicare AUC |
|-------|----------------|--------------|
| OCSVM | 0.951 | 0.526 |
| GMM | 0.959 | 0.598 |
| Isolation Forest | 0.950 | 0.608 |
| LOF | 0.953 | 0.604 |
| Best Ensemble | 0.955 | 0.598 |

---

## Key Graph Theory Concepts

1. Graph Representations (adjacency, Laplacian, bipartite, heterogeneous)
2. Centrality Measures (PageRank, betweenness, closeness, eigenvector)
3. Community Detection (modularity, Louvain, Girvan-Newman)
4. Spectral Graph Theory (Laplacian eigenvalues, Fiedler vector)
5. Graph Neural Networks (message passing framework)
6. Graph Anomaly Detection (DOMINANT, structural vs. attribute)
7. Motif Analysis (subgraph pattern counting)

---

## Author

**Baha Kirbasoglu** — Graph Theory Graduate Course Term Project
Companion project: [Fraud-Detection (ML)](../Fraud-Detection/) — BLG607 Data Mining
