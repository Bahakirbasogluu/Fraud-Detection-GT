"""
nb_03_classical_graph_analysis.py  (Colab + Drive edition)
===========================================================
Generates: notebooks/03_Classical_Graph_Analysis.ipynb
"""
import nbformat as nbf
from pathlib import Path
from colab_setup import DRIVE_MOUNT_CODE

NB_PATH = Path(__file__).parent.parent / "notebooks" / "03_Classical_Graph_Analysis.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": []},
}
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# 03 â€” Classical Graph Analysis
## Centrality, Community Detection & Motif Analysis

### Concepts:
1. **Centrality Measures** â€” PageRank, Betweenness, Closeness, Eigenvector
2. **Community Detection** â€” Louvain modularity optimization
3. **Motif Analysis** â€” Triangles, hubs, structural patterns
4. **Graph-Feature Classification** â€” centrality features â†’ Random Forest â†’ AUC

### Why This Matters:
Fraud nodes should differ structurally from normal nodes:
- Higher PageRank (suspicious neighborhoods reinforce each other)
- Lower clustering coefficient (radiating star patterns vs dense normal clusters)
- Concentrated in high-fraud communities (fraud rings)
"""))

cells.append(nbf.v4.new_code_cell(DRIVE_MOUNT_CODE))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt, matplotlib, time, json
import warnings; warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'figure.facecolor':'#0F0F1A','axes.facecolor':'#1A1A2E',
    'text.color':'#EEEEFF','axes.labelcolor':'#CCCCEE',
    'xtick.color':'#CCCCEE','ytick.color':'#CCCCEE',
})

from src.graph_builder import load_graph
from src.graph_features import (compute_centrality_features, detect_communities_louvain,
                                  community_fraud_analysis, motif_analysis)
from src.evaluation import compute_metrics

# Load graphs from Drive
G_cc  = load_graph(f'{GRAPHS_DIR}/credit_card/G_cc_knn.pkl')
G_med = load_graph(f'{GRAPHS_DIR}/medicare/G_med_provider.pkl')
print(f"âœ… Credit Card: {G_cc.number_of_nodes():,} nodes, {G_cc.number_of_edges():,} edges")
print(f"âœ… Medicare:    {G_med.number_of_nodes():,} nodes, {G_med.number_of_edges():,} edges")
"""))

# Centrality
cells.append(nbf.v4.new_markdown_cell("""\
---
## 1. Centrality Analysis

| Measure | Formula | Fraud Signal |
|---------|---------|--------------|
| **Degree** | $k(v) = |\\mathcal{N}(v)|$ | Fraud hubs have high degree |
| **PageRank** | $PR(v) = \\alpha\\sum_{u} \\frac{PR(u)}{k(u)}$ | Recursive importance |
| **Betweenness** | $\\frac{\\sigma(s,t|v)}{\\sigma(s,t)}$ | Broker in fraud ring |
| **Closeness** | $\\frac{n-1}{\\sum_u d(v,u)}$ | Distance efficiency |
| **Eigenvector** | $Ax = \\lambda x$ | Connected to well-connected |
| **Clustering** | $\\frac{2t(v)}{k(v)(k(v)-1)}$ | Dense local neighborhood |
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Computing Credit Card centrality features...")
t0 = time.time()
df_cc_cent = compute_centrality_features(G_cc)
print(f"  Done in {time.time()-t0:.1f}s")
df_cc_cent.to_csv(f'{METRICS_DIR}/cc_centrality_features.csv', index=False)

print("\\nComputing Medicare centrality features...")
t0 = time.time()
df_med_cent = compute_centrality_features(G_med)
print(f"  Done in {time.time()-t0:.1f}s")
df_med_cent.to_csv(f'{METRICS_DIR}/med_centrality_features.csv', index=False)

print("\\nâœ… Centrality features saved to Drive")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Centrality comparison plots
def plot_centrality_boxes(df, title, save_path):
    metrics = ['degree','pagerank','betweenness_centrality','closeness_centrality','clustering_coeff']
    metrics = [m for m in metrics if m in df.columns]
    df_v = df[df['label'].isin([0,1])].copy()

    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 5))
    fig.suptitle(title, fontsize=12, color='#EEEEFF', fontweight='bold', y=1.02)

    for ax, m in zip(axes, metrics):
        d_n = df_v[df_v['label']==0][m].values
        d_f = df_v[df_v['label']==1][m].values
        bp = ax.boxplot([d_n, d_f], labels=['Normal','Fraud'],
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('#4488FF88')
        bp['boxes'][1].set_facecolor('#FF444488')
        for med in bp['medians']: med.set_color('#FFFFFF')
        ax.set_title(m.replace('_','\\n'), fontsize=9); ax.grid(True, alpha=0.3)
        if len(d_f)>1:
            ps = np.sqrt((d_f.std()**2+d_n.std()**2)/2)
            d  = (d_f.mean()-d_n.mean())/(ps+1e-10)
            ax.set_xlabel(f"Cohen's d={d:.2f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_centrality_boxes(df_cc_cent,  'Credit Card: Centrality by Fraud Label',
                       f'{FIGURES_DIR}/cc_centrality_comparison.png')
plot_centrality_boxes(df_med_cent, 'Medicare: Centrality by Fraud Label',
                       f'{FIGURES_DIR}/med_centrality_comparison.png')
"""))

# Community Detection
cells.append(nbf.v4.new_markdown_cell("""\
---
## 2. Community Detection â€” Louvain Algorithm

Optimizes **modularity** $Q$:
$$Q = \\frac{1}{2m} \\sum_{ij} \\left[ A_{ij} - \\frac{k_i k_j}{2m} \\right] \\delta(c_i, c_j)$$

High $Q$ â†’ strong community structure â†’ fraud rings appear as dense sub-communities.
"""))

cells.append(nbf.v4.new_code_cell("""\
t0 = time.time()
partition_cc = detect_communities_louvain(G_cc, seed=42)
print(f"Credit Card communities: {len(set(partition_cc.values()))} (in {time.time()-t0:.1f}s)")
comm_cc = community_fraud_analysis(G_cc, partition_cc)
print(comm_cc.head(10).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
t0 = time.time()
partition_med = detect_communities_louvain(G_med, seed=42)
print(f"Medicare communities: {len(set(partition_med.values()))} (in {time.time()-t0:.1f}s)")
comm_med = community_fraud_analysis(G_med, partition_med)
print(comm_med.head(10).to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Community fraud rate visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Community Detection: Fraud Concentration per Community',
              fontsize=13, color='#EEEEFF', fontweight='bold')

for ax, (df_c, title) in zip(axes, [(comm_cc,'Credit Card'), (comm_med,'Medicare')]):
    top = df_c.nlargest(30, 'fraud_rate')
    colors = ['#FF4444' if r > 0.05 else '#5577FF' for r in top['fraud_rate']]
    ax.bar(range(len(top)), top['fraud_rate']*100, color=colors, alpha=0.85, edgecolor='#333')
    mean_r = df_c['fraud_rate'].mean()*100
    ax.axhline(mean_r, color='#FFD700', linestyle='--', alpha=0.7, label=f'Mean: {mean_r:.2f}%')
    ax.set_xlabel('Community (sorted by fraud rate)')
    ax.set_ylabel('Fraud Rate (%)'); ax.set_title(title)
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/community_fraud_rates.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Motif Analysis
cells.append(nbf.v4.new_markdown_cell("""\
---
## 3. Motif Analysis â€” Triangles & Structural Patterns

**Triangles:** A â†’ B â†’ C â†’ A  
Fraud rings often form complete triads (mutual trust); ghost-billing uses star topologies.

**Hub detection:** Nodes with degree > avg + 2Ïƒ are potential fraud coordinators.
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Motif analysis â€” Credit Card...")
motif_cc = motif_analysis(G_cc)

print("\\nMotif analysis â€” Medicare...")
motif_med = motif_analysis(G_med)

# Triangle count: fraud vs normal
def plot_triangles(G, motif_data, title, save_path):
    tri = motif_data['triangle_per_node']
    f_tri = [tri.get(n,0) for n in G.nodes() if G.nodes[n].get('label')==1]
    n_tri = [tri.get(n,0) for n in G.nodes() if G.nodes[n].get('label')==0]

    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(n_tri, bins=40, alpha=0.6, color='#4488FF', label=f'Normal', density=True)
    if f_tri: ax.hist(f_tri, bins=40, alpha=0.8, color='#FF4444', label=f'Fraud', density=True)
    ax.set_title(title, color='#EEEEFF'); ax.set_xlabel('Triangle Count')
    ax.set_ylabel('Density'); ax.set_yscale('log'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.show()
    if f_tri:
        print(f"  Mean triangles â€” Fraud: {np.mean(f_tri):.2f}, Normal: {np.mean(n_tri):.2f}")

plot_triangles(G_cc,  motif_cc,  'Credit Card: Triangle Count',
               f'{FIGURES_DIR}/cc_triangles.png')
plot_triangles(G_med, motif_med, 'Medicare: Triangle Count',
               f'{FIGURES_DIR}/med_triangles.png')
"""))

# Graph Feature Classification
cells.append(nbf.v4.new_markdown_cell("""\
---
## 4. Graph-Feature Classification (Category: Graph-Classical)

Use centrality measures as ML features â†’ Random Forest â†’ AUC on test set.
This gives us the **"Graph-Classical"** row in the final comparison table.
"""))

cells.append(nbf.v4.new_code_cell("""\
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

graph_classical_results = []

def classify_with_centrality(df_cent, dataset_name):
    feat_cols = ['degree','degree_centrality','pagerank','betweenness_centrality',
                 'closeness_centrality','eigenvector_centrality','clustering_coeff']
    feat_cols = [c for c in feat_cols if c in df_cent.columns]
    df_v = df_cent[df_cent['label'].isin([0,1])].dropna(subset=feat_cols)

    if df_v['label'].sum() == 0:
        print(f"  {dataset_name}: No fraud labels â€” skipping")
        return

    X = StandardScaler().fit_transform(df_v[feat_cols].values)
    y = df_v['label'].values
    print(f"\\n{dataset_name}: {len(X):,} nodes | {y.sum():,} fraud ({y.mean()*100:.2f}%)")

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    scores = cross_val_predict(rf, X, y, cv=5, method='predict_proba')[:,1]
    auc = roc_auc_score(y, scores)
    ap  = average_precision_score(y, scores)
    print(f"  RandomForest on Centrality â†’ AUC: {auc:.4f} | AUPRC: {ap:.4f}")

    graph_classical_results.append({
        'dataset': dataset_name, 'category': 'Graph-Classical',
        'method': 'RF_Centrality_Features',
        'auc_roc': auc, 'auc_prc': ap, 'f1': None
    })

classify_with_centrality(df_cc_cent,  'Credit Card')
classify_with_centrality(df_med_cent, 'Medicare')

with open(f'{METRICS_DIR}/classical_graph_results.json', 'w') as f:
    json.dump(graph_classical_results, f, indent=2, default=str)
print("\\nâœ… Saved to Drive:", f'{METRICS_DIR}/classical_graph_results.json')
"""))

cells.append(nbf.v4.new_code_cell("""\
print("=" * 55)
print("  NOTEBOOK 03 â€” COMPLETE")
print("=" * 55)
for r in graph_classical_results:
    print(f"  {r['dataset']:<15} | {r['method']}: AUC={r['auc_roc']:.4f}")
print()
print("  All figures + metrics saved to Drive âœ…")
print("  Next â†’ 04_Spectral_Methods.ipynb")
print("=" * 55)
"""))

nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"âœ… Generated: {NB_PATH}")

