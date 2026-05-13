"""
nb_02_graph_construction.py  â€” final Colab edition
"""
import nbformat as nbf
from pathlib import Path
from colab_setup import DRIVE_MOUNT_CODE

NB_PATH = Path(__file__).parent.parent / "notebooks" / "02_Graph_Construction.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": []},
}
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# 02 â€” Graph Construction
## From Tabular Data to Graph Representations

| Dataset | Graph | Nodes | Edges |
|---------|-------|-------|-------|
| Credit Card | K-NN Similarity | Transactions | Cosine similarity |
| Medicare | Bipartite Projection | Providers | Shared drug patterns |

**Medicare pipeline this notebook:**
```
3x CMS CSVs  +  LEIE UPDATED.csv
     |                |
     v                v
 concat (3yr)    excluded NPIs
         \\          /
          join on NPI --> fraud_label
               |
          build B(P,D)  [bipartite: providers <-> drugs]
               |
          project --> G(P,P)  [provider-provider, edge = shared drugs]
```
"""))

cells.append(nbf.v4.new_code_cell(DRIVE_MOUNT_CODE))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np, pandas as pd, networkx as nx
import matplotlib.pyplot as plt, matplotlib, warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'figure.facecolor':'#0F0F1A', 'axes.facecolor':'#1A1A2E',
    'text.color':'#EEEEFF', 'axes.labelcolor':'#CCCCEE',
    'xtick.color':'#CCCCEE', 'ytick.color':'#CCCCEE',
})
from src.graph_builder import (CreditCardGraphBuilder, MedicareGraphBuilder,
                                 save_graph, graph_summary)
print("Setup complete")
"""))

# CC graph
cells.append(nbf.v4.new_markdown_cell("""\
---
## 1. Credit Card: K-NN Similarity Graph

Every transaction = node.
Edges connect the k most similar transactions in PCA feature space (cosine similarity).
"""))

cells.append(nbf.v4.new_code_cell("""\
cc_builder = CreditCardGraphBuilder(CONFIG)
G_cc = cc_builder.build(CC_PATH)

s = graph_summary(G_cc)
print("\\nCredit Card Graph:")
for k,v in s.items(): print(f"  {k:20s}: {v}")

save_graph(G_cc, f'{GRAPHS_DIR}/credit_card/G_cc_knn.pkl')
"""))

cells.append(nbf.v4.new_code_cell("""\
# Fraud ego-subgraph visualization
import random; random.seed(42)
fraud_nodes = [n for n in G_cc.nodes() if G_cc.nodes[n].get('label')==1]
sample_fn   = random.sample(fraud_nodes, min(25, len(fraud_nodes)))
sub_nodes   = set(sample_fn)
for fn in sample_fn: sub_nodes.update(list(G_cc.neighbors(fn))[:4])
G_sub = G_cc.subgraph(list(sub_nodes))

fig, ax = plt.subplots(figsize=(10,8))
ax.set_facecolor('#0F0F1A'); fig.patch.set_facecolor('#0F0F1A')
pos = nx.spring_layout(G_sub, seed=42, k=0.6)
nc  = ['#FF4444' if G_cc.nodes[n].get('label')==1 else '#4488FF' for n in G_sub.nodes()]
ns  = [250 if G_cc.nodes[n].get('label')==1 else 40 for n in G_sub.nodes()]
nx.draw_networkx_nodes(G_sub,pos,node_color=nc,node_size=ns,alpha=0.9,ax=ax)
nx.draw_networkx_edges(G_sub,pos,alpha=0.25,edge_color='#8888BB',width=0.5,ax=ax)

from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#FF4444',markersize=10,label='Fraud'),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='#4488FF',markersize=8, label='Normal'),
], loc='upper right', facecolor='#1A1A2E', edgecolor='#555')
ax.set_title('Credit Card: Fraud Ego-Subgraph', color='#EEEEFF', fontsize=12)
ax.axis('off'); plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/cc_fraud_subgraph.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# Medicare graph
cells.append(nbf.v4.new_markdown_cell("""\
---
## 2. Medicare: Provider-Drug Bipartite -> Provider-Provider Projection

**Data sources:**
- `medicare/` folder: 3 CSV files (2017, 2018, 2019) â€” auto-loaded
- `leie/UPDATED.csv`: OIG exclusion list â€” joined on NPI

**Memory note:** Full 3-year Medicare is ~4.5 GB.
The builder subsamples to `max_providers` (config: 30,000) automatically.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Build Medicare graph
# MED_DIR  = directory with the 3 Medicare CSVs
# LEIE_PATH = path to UPDATED.csv
med_builder = MedicareGraphBuilder(CONFIG)

G_med, B_med, df_clean, col_info = med_builder.build(
    med_dir   = MED_DIR,
    leie_path = LEIE_PATH,
)

s_med = graph_summary(G_med)
print("\\nMedicare Provider Graph:")
for k,v in s_med.items(): print(f"  {k:20s}: {v}")

save_graph(G_med, f'{GRAPHS_DIR}/medicare/G_med_provider.pkl')
save_graph(B_med, f'{GRAPHS_DIR}/medicare/B_med_bipartite.pkl')
print(f"\\nSaved both graphs to {GRAPHS_DIR}/medicare/")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Stats and plots
prov_nodes = [n for n,d in B_med.nodes(data=True) if d.get('bipartite')==0]
drug_nodes = [n for n,d in B_med.nodes(data=True) if d.get('bipartite')==1]
fraud_prov = [n for n in prov_nodes if B_med.nodes[n].get('label')==1]
print(f"Bipartite graph:")
print(f"  Providers : {len(prov_nodes):,}")
print(f"  Drugs     : {len(drug_nodes):,}")
print(f"  Edges     : {B_med.number_of_edges():,}")
print(f"  Fraud prov: {len(fraud_prov):,} ({len(fraud_prov)/max(len(prov_nodes),1)*100:.2f}%)")

# Degree comparison
frd_deg = [G_med.degree(n) for n in G_med.nodes() if G_med.nodes[n].get('label')==1]
nrm_deg = [G_med.degree(n) for n in G_med.nodes() if G_med.nodes[n].get('label')==0]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Medicare Provider Graph Analysis', fontsize=13, color='#EEEEFF', fontweight='bold')

ax = axes[0]
if frd_deg:
    ax.hist(nrm_deg, bins=50, alpha=0.6, color='#4488FF',
            label=f'Normal ({len(nrm_deg):,})', density=True)
    ax.hist(frd_deg, bins=50, alpha=0.8, color='#FF4444',
            label=f'Fraud  ({len(frd_deg):,})', density=True)
    print(f"Avg degree  Fraud: {np.mean(frd_deg):.1f}  Normal: {np.mean(nrm_deg):.1f}")
else:
    ax.text(0.5,0.5,'No fraud labels in graph sample',
            ha='center',va='center',transform=ax.transAxes,color='#EEEEFF')
ax.set_title('Provider Degree by Fraud Label'); ax.set_xlabel('Degree')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
weights=[d.get('weight',1) for _,_,d in G_med.edges(data=True)]
ax.hist(weights, bins=50, color='#7788FF', alpha=0.85, edgecolor='#333')
ax.set_xlabel('Edge Weight (shared drugs)'); ax.set_ylabel('Count')
ax.set_title('Edge Weight Distribution'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/medicare_graph_construction.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
print("=" * 58)
print("  NOTEBOOK 02 COMPLETE")
print("=" * 58)
print(f"  CC  Graph : {s['n_nodes']:,} nodes | {s['n_edges']:,} edges | fraud={s['fraud_rate']*100:.2f}%")
print(f"  Med Graph : {s_med['n_nodes']:,} nodes | {s_med['n_edges']:,} edges | fraud={s_med['fraud_rate']*100:.2f}%")
print("  Graphs saved to Drive")
print("  Next -> 03_Classical_Graph_Analysis.ipynb")
print("=" * 58)
"""))

nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Generated: {}".format(NB_PATH))

