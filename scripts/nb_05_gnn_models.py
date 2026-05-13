"""
nb_05_gnn_models.py  (Colab + Drive edition)
=============================================
Generates: notebooks/05_GNN_Models.ipynb
"""
import nbformat as nbf
from pathlib import Path
from colab_setup import DRIVE_MOUNT_CODE, INSTALL_CODE

NB_PATH = Path(__file__).parent.parent / "notebooks" / "05_GNN_Models.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": [], "gpuType": "T4"},
    "accelerator": "GPU",
}
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# 05 â€” Graph Neural Network Models
## GCN, GraphSAGE, GAT, DOMINANT on Fraud Graphs

> **Colab tip:** Enable GPU: Runtime â†’ Change runtime type â†’ T4 GPU

| Model | Architecture | Reference |
|-------|-------------|-----------|
| **GCN** | $H^{(l+1)}=\sigma(\hat{A}H^{(l)}W^{(l)})$ | Kipf & Welling, 2017 |
| **GraphSAGE** | Mean/Max neighborhood sampling | Hamilton et al., 2017 |
| **GAT** | Attention-weighted aggregation | VeliÄkoviÄ‡ et al., 2018 |
| **DOMINANT** | Attribute + structure autoencoder | Ding et al., 2019 |

All models input: **centrality features** (computed in Notebook 03) + graph structure.
"""))

cells.append(nbf.v4.new_code_cell(DRIVE_MOUNT_CODE))

# Package install
cells.append(nbf.v4.new_code_cell("""\
# Install PyTorch Geometric (Colab already has PyTorch)
import subprocess, sys
print("Installing PyTorch Geometric...")
subprocess.check_call([sys.executable,"-m","pip","install","-q","torch-geometric"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","networkx","pyyaml","pyvis"])
print("âœ… Packages installed")
"""))

cells.append(nbf.v4.new_code_cell("""\
import torch, numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt, matplotlib, time, json
import warnings; warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'figure.facecolor':'#0F0F1A','axes.facecolor':'#1A1A2E',
    'text.color':'#EEEEFF','axes.labelcolor':'#CCCCEE',
    'xtick.color':'#CCCCEE','ytick.color':'#CCCCEE',
})

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"âœ… PyTorch: {torch.__version__}")
print(f"   Device:  {device}")
if device == 'cuda':
    print(f"   GPU:     {torch.cuda.get_device_name(0)}")

import torch_geometric
print(f"âœ… PyG: {torch_geometric.__version__}")

from src.graph_builder import load_graph
from src.gnn_models import GCN, GraphSAGE, GAT, DOMINANT, GNNTrainer, prepate_pyg_data
from src.evaluation import compute_metrics
"""))

# Load data
cells.append(nbf.v4.new_code_cell("""\
# Load graphs from Drive
G_cc  = load_graph(f'{GRAPHS_DIR}/credit_card/G_cc_knn.pkl')
G_med = load_graph(f'{GRAPHS_DIR}/medicare/G_med_provider.pkl')
print(f"Graphs loaded âœ…")

# Load centrality features (cached in Notebook 03)
try:
    df_cc_cent  = pd.read_csv(f'{METRICS_DIR}/cc_centrality_features.csv')
    df_med_cent = pd.read_csv(f'{METRICS_DIR}/med_centrality_features.csv')
    print(f"Centrality features loaded âœ…")
except FileNotFoundError:
    print("âš ï¸  Centrality features not found â€” run Notebook 03 first!")
    print("  Computing now (may take 10+ min)...")
    from src.graph_features import compute_centrality_features
    df_cc_cent  = compute_centrality_features(G_cc,  verbose=True)
    df_med_cent = compute_centrality_features(G_med, verbose=True)
    df_cc_cent.to_csv(f'{METRICS_DIR}/cc_centrality_features.csv', index=False)
    df_med_cent.to_csv(f'{METRICS_DIR}/med_centrality_features.csv', index=False)

print(f"CC  features: {df_cc_cent.shape}")
print(f"Med features: {df_med_cent.shape}")
"""))

# Prepare PyG data
cells.append(nbf.v4.new_markdown_cell("---\n## 1. Prepare PyTorch Geometric Data Objects"))

cells.append(nbf.v4.new_code_cell("""\
data_cc  = prepate_pyg_data(G_cc,  df_cc_cent,  seed=42, train_ratio=0.70, val_ratio=0.10)
data_med = prepate_pyg_data(G_med, df_med_cent, seed=42, train_ratio=0.70, val_ratio=0.10)
print(f"\\nCC  Data: {data_cc}")
print(f"Med Data: {data_med}")
"""))

# Training helper
cells.append(nbf.v4.new_code_cell("""\
gnn_results = []

def train_and_eval(model, data, model_name, dataset_name, epochs=200, lr=0.01):
    trainer = GNNTrainer(model, {}, device=device)
    print(f"\\n{'='*50}")
    print(f"Training {model_name} on {dataset_name}...")
    t0 = time.time()
    history = trainer.fit(data, epochs=epochs, lr=lr, verbose=True)
    train_time = time.time()-t0

    test_m = trainer.evaluate(data, 'test_mask')
    print(f"\\nğŸ“Š {model_name} | {dataset_name} | TEST RESULTS:")
    print(f"   AUC-ROC : {test_m['auc_roc']:.4f}")
    print(f"   AUC-PRC : {test_m['auc_prc']:.4f}")
    print(f"   F1      : {test_m['f1']:.4f}")
    print(f"   Time    : {train_time:.1f}s")

    # Save model to Drive
    save_p = f'{MODELS_DIR}/{model_name}_{dataset_name.replace(" ","_").lower()}.pt'
    torch.save(model.state_dict(), save_p)
    print(f"   Saved   : {save_p}")

    gnn_results.append({
        'dataset': dataset_name, 'category': 'Graph-GNN',
        'method': model_name,
        'auc_roc': test_m['auc_roc'], 'auc_prc': test_m['auc_prc'],
        'f1': test_m['f1'], 'train_time': train_time,
    })
    return history
"""))

# GCN
cells.append(nbf.v4.new_markdown_cell("""\
---
## 2. GCN â€” Graph Convolutional Network
```
GCNConv(in â†’ 64) â†’ ReLU â†’ Dropout(0.5)
GCNConv(64 â†’ 32) â†’ ReLU â†’ Dropout(0.5)
Linear(32 â†’ 1)   â†’ Sigmoid â†’ fraud probability
```
"""))

cells.append(nbf.v4.new_code_cell("""\
in_cc  = data_cc.x.shape[1]
in_med = data_med.x.shape[1]

# Credit Card
gcn_cc = GCN(in_cc, hidden_channels=64, dropout=0.5)
print(gcn_cc)
hist_gcn_cc = train_and_eval(gcn_cc, data_cc, 'GCN', 'Credit Card', epochs=200, lr=0.01)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Medicare
gcn_med = GCN(in_med, hidden_channels=64, dropout=0.5)
hist_gcn_med = train_and_eval(gcn_med, data_med, 'GCN', 'Medicare', epochs=200, lr=0.01)
"""))

# GraphSAGE
cells.append(nbf.v4.new_markdown_cell("---\n## 3. GraphSAGE â€” Inductive Representation Learning"))

cells.append(nbf.v4.new_code_cell("""\
sage_cc  = GraphSAGE(in_cc,  hidden_channels=64, aggr='mean', dropout=0.5)
sage_med = GraphSAGE(in_med, hidden_channels=64, aggr='mean', dropout=0.5)

hist_sage_cc  = train_and_eval(sage_cc,  data_cc,  'GraphSAGE', 'Credit Card', epochs=200, lr=0.01)
hist_sage_med = train_and_eval(sage_med, data_med, 'GraphSAGE', 'Medicare',    epochs=200, lr=0.01)
"""))

# GAT
cells.append(nbf.v4.new_markdown_cell("---\n## 4. GAT â€” Graph Attention Network"))

cells.append(nbf.v4.new_code_cell("""\
gat_cc  = GAT(in_cc,  hidden_channels=8, heads=4, dropout=0.6)
gat_med = GAT(in_med, hidden_channels=8, heads=4, dropout=0.6)

hist_gat_cc  = train_and_eval(gat_cc,  data_cc,  'GAT', 'Credit Card', epochs=200, lr=0.005)
hist_gat_med = train_and_eval(gat_med, data_med, 'GAT', 'Medicare',    epochs=200, lr=0.005)
"""))

# DOMINANT
cells.append(nbf.v4.new_markdown_cell("""\
---
## 5. DOMINANT â€” Unsupervised Deep Anomaly Detection

No labels during training. Learns to reconstruct node attributes and graph structure.
Anomaly score: $s(v) = \\alpha \\cdot err_{struct}(v) + (1-\\alpha) \\cdot err_{attr}(v)$
"""))

cells.append(nbf.v4.new_code_cell("""\
def train_dominant(data, dataset_name, epochs=100, lr=0.001, alpha=0.5):
    import torch.nn as nn
    model = DOMINANT(data.x.shape[1], hidden_channels=64,
                     embedding_dim=32, dropout=0.3, alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    data_d = data.to(device)
    n = data_d.x.shape[0]

    print(f"\\nTraining DOMINANT on {dataset_name} ({epochs} epochs)...")
    t0 = time.time()
    for epoch in range(1, epochs+1):
        model.train(); optimizer.zero_grad()
        x_hat, adj_hat, z = model(data_d.x, data_d.edge_index)
        adj_dense = torch.zeros(n, n, device=device)
        adj_dense[data_d.edge_index[0], data_d.edge_index[1]] = 1.0
        attr_loss   = ((data_d.x - x_hat)**2).mean()
        struct_loss = ((adj_dense - adj_hat)**2).mean()
        loss = alpha*struct_loss + (1-alpha)*attr_loss
        loss.backward(); optimizer.step()
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Loss:{loss.item():.4f} "
                  f"Attr:{attr_loss.item():.4f} Struct:{struct_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_hat, adj_hat, z = model(data_d.x, data_d.edge_index)
        adj_dense = torch.zeros(n, n, device=device)
        adj_dense[data_d.edge_index[0], data_d.edge_index[1]] = 1.0
        scores = model.compute_anomaly_scores(data_d.x, x_hat, adj_hat, adj_dense).cpu().numpy()

    labels = data_d.y.cpu().numpy()
    test_mask = data_d.test_mask.cpu().numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    if labels[test_mask].sum() > 0:
        auc = roc_auc_score(labels[test_mask], scores[test_mask])
        ap  = average_precision_score(labels[test_mask], scores[test_mask])
        print(f"\\n  DOMINANT | {dataset_name} â†’ AUC:{auc:.4f} | AP:{ap:.4f} ({time.time()-t0:.0f}s)")
    else:
        auc, ap = 0, 0

    gnn_results.append({'dataset': dataset_name, 'category': 'Graph-GNN-Unsup',
                         'method': 'DOMINANT', 'auc_roc': auc, 'auc_prc': ap, 'f1': None,
                         'train_time': time.time()-t0})

train_dominant(data_cc,  'Credit Card', epochs=100)
train_dominant(data_med, 'Medicare',    epochs=100)
"""))

# Training curves
cells.append(nbf.v4.new_code_cell("""\
# Training curves
def plot_training_curves(histories, labels, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    fig.suptitle(title, fontsize=12, color='#EEEEFF')
    colors = ['#5577FF','#FF5577','#55DD88','#FFAA33']

    for ax, (metric, ylabel) in zip(axes, [('train_loss','Training Loss'),('auc_roc','Val AUC-ROC')]):
        for hist, lbl, col in zip(histories, labels, colors):
            if not hist: continue
            epochs = [h['epoch'] for h in hist]
            vals   = [h.get(metric, 0) for h in hist]
            ax.plot(epochs, vals, color=col, lw=2, label=lbl)
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

plot_training_curves([hist_gcn_cc, hist_sage_cc, hist_gat_cc],
                     ['GCN','GraphSAGE','GAT'],
                     'Credit Card: GNN Training Curves',
                     f'{FIGURES_DIR}/cc_gnn_training.png')

plot_training_curves([hist_gcn_med, hist_sage_med, hist_gat_med],
                     ['GCN','GraphSAGE','GAT'],
                     'Medicare: GNN Training Curves',
                     f'{FIGURES_DIR}/med_gnn_training.png')
"""))

# Save results
cells.append(nbf.v4.new_code_cell("""\
with open(f'{METRICS_DIR}/gnn_results.json', 'w') as f:
    json.dump(gnn_results, f, indent=2, default=str)
print("âœ… GNN results saved to Drive")

print("\\n" + "="*55)
print("  NOTEBOOK 05 â€” COMPLETE")
print("="*55)
for r in gnn_results:
    print(f"  {r['dataset']:<15}|{r['method']:<12}: AUC={r['auc_roc']:.4f} AP={r['auc_prc']:.4f}")
print()
print("  Models + results saved to Drive âœ…")
print("  Next â†’ 06_Comparison_and_Results.ipynb")
print("="*55)
"""))

nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Generated: {}".format(NB_PATH))


