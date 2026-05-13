"""
nb_01_eda_baseline.py  â€” final Colab edition
"""
import nbformat as nbf
from pathlib import Path
from colab_setup import DRIVE_MOUNT_CODE, INSTALL_CODE

NB_PATH = Path(__file__).parent.parent / "notebooks" / "01_EDA_and_Baseline.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": []},
}
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# 01 â€” EDA & ML Baseline
## Fraud Detection via Graph Theory | Graduate Graph Theory Course

**Research Hypothesis:**
> Fraudulent entities form structurally distinct communities in transaction graphs.
> Graph-theoretic methods significantly outperform tabular anomaly detection
> on relational fraud data â€” especially for Medicare provider-level fraud.

**Drive structure required:**
```
MyDrive/fraudDataset/
  creditcard / creditcard.csv
  leie       / UPDATED.csv
  medicare   / Medicare_Part_D_...2017.csv
  medicare   / Medicare_Part_D_...2018.csv
  medicare   / Medicare_Part_D_...2019.csv
  Fraud-Detection-GT / (this project)
```
"""))

cells.append(nbf.v4.new_code_cell(DRIVE_MOUNT_CODE))
cells.append(nbf.v4.new_code_cell(INSTALL_CODE))

cells.append(nbf.v4.new_code_cell("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib, seaborn as sns, warnings, json
warnings.filterwarnings('ignore')
matplotlib.rcParams.update({
    'figure.facecolor':'#0F0F1A', 'axes.facecolor':'#1A1A2E',
    'text.color':'#EEEEFF', 'axes.labelcolor':'#CCCCEE',
    'xtick.color':'#CCCCEE', 'ytick.color':'#CCCCEE',
})
from src.baseline_loader import HARDCODED_BASELINE
from src.evaluation import get_ml_baseline_table
print("Libraries loaded OK")
"""))

# â”€â”€ Credit Card EDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cells.append(nbf.v4.new_markdown_cell("""\
---
## 1. Credit Card Fraud Dataset
- 284,807 transactions | 492 fraud (0.172%)
- 28 PCA features + Amount + Time
"""))

cells.append(nbf.v4.new_code_cell("""\
try:
    df_cc = pd.read_csv(CC_PATH)
    print(f"Loaded: {df_cc.shape} | Fraud: {df_cc['Class'].sum()} ({df_cc['Class'].mean()*100:.3f}%)")
except FileNotFoundError:
    print("creditcard.csv not found -- using synthetic demo")
    np.random.seed(42); n=5000
    df_cc = pd.DataFrame(np.random.randn(n,28), columns=[f'V{i}' for i in range(1,29)])
    df_cc['Amount']=(np.abs(np.random.exponential(100,n)))
    df_cc['Time']=np.arange(n)
    df_cc['Class']=(np.random.rand(n)<0.002).astype(int)
df_cc.head(3)
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle('Credit Card Fraud Dataset -- Overview', fontsize=13, color='#EEEEFF', fontweight='bold')

ax = axes[0]
counts = df_cc['Class'].value_counts()
b = ax.bar(['Normal','Fraud'], counts.values, color=['#4488FF','#FF4444'], alpha=0.85, edgecolor='#333')
ax.set_title('Class Distribution'); ax.set_ylabel('Count'); ax.grid(True, axis='y', alpha=0.3)
for bar,v in zip(b,counts.values): ax.text(bar.get_x()+bar.get_width()/2, v+200, f'{v:,}', ha='center')

ax = axes[1]
for cls,col,lbl in [(0,'#4488FF','Normal'),(1,'#FF4444','Fraud')]:
    d=df_cc[df_cc['Class']==cls]['Amount']; ax.hist(d[d<500],bins=50,alpha=0.6,color=col,label=lbl,density=True)
ax.set_title('Amount (< $500)'); ax.set_xlabel('Amount ($)'); ax.legend(); ax.grid(True,alpha=0.3)

ax = axes[2]
fcols=[f'V{i}' for i in range(1,15)]
ax.barh(fcols,[abs(df_cc[c].corr(df_cc['Class'])) for c in fcols],color='#7788FF',alpha=0.85,edgecolor='#333')
ax.set_title('Feature Correlation with Fraud'); ax.set_xlabel('|Pearson|'); ax.grid(True,axis='x',alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/cc_eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

# â”€â”€ Medicare EDA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cells.append(nbf.v4.new_markdown_cell("""\
---
## 2. Medicare + LEIE Datasets

**Medicare Part D (CMS):** Provider-level prescription data 2017-2019 (3 files)
**LEIE (OIG):** Excluded providers list -- our fraud ground truth

**Join logic:** If a provider's NPI appears in LEIE -> label = 1 (fraud)
"""))

cells.append(nbf.v4.new_code_cell("""\
import glob, os

# Load Medicare (sample 100k rows from each file to avoid memory issues in EDA)
print(f"Medicare files found: {len(MED_FILES)}")
for f in MED_FILES: print(f"  {os.path.basename(f)}")

med_dfs = []
for fp in MED_FILES:
    yr = next((y for y in ['2017','2018','2019'] if y in fp), None)
    df_tmp = pd.read_csv(fp, nrows=100000, low_memory=False)
    if yr: df_tmp['year'] = yr
    med_dfs.append(df_tmp)
    print(f"  Loaded {os.path.basename(fp)}: {df_tmp.shape}")

df_med_sample = pd.concat(med_dfs, ignore_index=True)
print(f"\\nCombined sample: {df_med_sample.shape}")
print(f"Columns: {df_med_sample.columns.tolist()[:15]}...")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Load LEIE
print(f"Loading LEIE from: {LEIE_PATH}")
leie_df = pd.read_csv(LEIE_PATH, low_memory=False)
print(f"LEIE shape: {leie_df.shape}")
print(f"LEIE columns: {leie_df.columns.tolist()}")

# Find NPI column
leie_npi_col = next((c for c in leie_df.columns if 'NPI' in c.upper() or 'npi' in c.lower()), None)
print(f"\\nNPI column: {leie_npi_col}")

if leie_npi_col:
    valid_npis = leie_df[leie_npi_col].astype(str).str.strip()
    valid_npis = set(n for n in valid_npis if n.isdigit() and len(n)==10 and n!='0000000000')
    print(f"Valid 10-digit NPIs in LEIE: {len(valid_npis):,}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Medicare NPI column detection
npi_candidates = ['Prscrbr_NPI','prscrbr_npi','NPI','npi']
drug_candidates = ['Gnrc_Name','gnrc_name','Brnd_Name','brnd_name']

npi_col_med  = next((c for c in df_med_sample.columns if c.lower() in [x.lower() for x in npi_candidates]), None)
drug_col_med = next((c for c in df_med_sample.columns if c.lower() in [x.lower() for x in drug_candidates]), None)
state_col_med = next((c for c in df_med_sample.columns
                       if 'state' in c.lower() and 'abrvtn' in c.lower()), None)
claims_col_med = next((c for c in df_med_sample.columns if 'clm' in c.lower()), None)

print(f"NPI col:    {npi_col_med}")
print(f"Drug col:   {drug_col_med}")
print(f"State col:  {state_col_med}")
print(f"Claims col: {claims_col_med}")

if npi_col_med and leie_npi_col:
    df_med_sample['_npi_str'] = df_med_sample[npi_col_med].astype(str).str.strip()
    df_med_sample['fraud_label'] = df_med_sample['_npi_str'].isin(valid_npis).astype(int)
    fraud_rate = df_med_sample.drop_duplicates(npi_col_med)['fraud_label'].mean()
    print(f"\\nFraud rate (unique providers): {fraud_rate*100:.3f}%")
    print(f"Fraud providers in sample: {df_med_sample.drop_duplicates(npi_col_med)['fraud_label'].sum():,}")
"""))

cells.append(nbf.v4.new_code_cell("""\
# Medicare EDA plots
if npi_col_med and drug_col_med:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('Medicare Dataset -- Sample Overview', fontsize=13, color='#EEEEFF', fontweight='bold')

    # Year distribution
    ax = axes[0]
    if 'year' in df_med_sample.columns:
        vc = df_med_sample['year'].value_counts().sort_index()
        ax.bar(vc.index.astype(str), vc.values, color=['#5577FF','#FF7755','#55DD88'], alpha=0.85, edgecolor='#333')
        ax.set_xlabel('Year'); ax.set_ylabel('Rows'); ax.set_title('Records per Year')
        ax.grid(True, axis='y', alpha=0.3)

    # State fraud rate (if available)
    ax = axes[1]
    if state_col_med and 'fraud_label' in df_med_sample.columns:
        by_state = (df_med_sample.groupby(state_col_med)['fraud_label']
                    .mean().nlargest(15)*100)
        ax.barh(by_state.index, by_state.values, color='#FF7755', alpha=0.85, edgecolor='#333')
        ax.set_xlabel('Fraud Rate (%)'); ax.set_title('Fraud Rate by State (Top 15)')
        ax.grid(True, axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'State/label data not available',
                ha='center', va='center', transform=ax.transAxes, color='#EEEEFF')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/medicare_eda_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("Could not identify required columns for EDA plots")
"""))

# â”€â”€ ML Baseline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
cells.append(nbf.v4.new_markdown_cell("""\
---
## 3. ML Baseline (from BLG607 Companion Project)

| Dataset | Best Model | AUC-ROC | Note |
|---------|-----------|---------|------|
| Credit Card | GMM | **0.959** | High â€” less relational |
| Medicare | IsoForest | **0.608** | Near-random â€” graph should dominate |
"""))

cells.append(nbf.v4.new_code_cell("""\
try:
    ml_df = get_ml_baseline_table(ML_PICKLES_DIR)
    print("ML results loaded from Drive")
except Exception:
    print("Using hardcoded baseline (pickles not on Drive)")
    recs = []
    for ds, methods in HARDCODED_BASELINE.items():
        for m, v in methods.items():
            recs.append({'Dataset':ds,'Method':m,'AUC_ROC':v['AUC_ROC'],
                         'AUC_PRC':v['AUC_PRC'],'Category':'Tabular-ML'})
    ml_df = pd.DataFrame(recs)

print(ml_df.to_string(index=False))
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ML Baseline: Tabular Methods\\n(from BLG607 Data Mining Project)',
              fontsize=13, color='#EEEEFF', fontweight='bold')

for ax, (ds, kw) in zip(axes, [('Credit Card','Credit'),('Medicare','Medicare')]):
    sub = ml_df[ml_df['Dataset'].str.contains(kw, na=False)]
    if len(sub)==0:
        dk = ds; sub = pd.DataFrame(
            [{'Method':m,'AUC_ROC':v['AUC_ROC']} for m,v in HARDCODED_BASELINE[dk].items()])
    aucs = sub['AUC_ROC'].values; methods = sub['Method'].values
    si = np.argsort(aucs)
    bars = ax.barh(methods[si], aucs[si],
                   color=['#FF4444' if a==max(aucs) else '#5577FF' for a in aucs[si]],
                   alpha=0.85, edgecolor='#333')
    ax.set_xlim(0,1.08); ax.set_xlabel('AUC-ROC'); ax.set_title(ds)
    ax.axvline(0.5,color='gray',linestyle='--',alpha=0.4)
    for bar,v in zip(bars,aucs[si]):
        ax.text(v+0.005,bar.get_y()+bar.get_height()/2,f'{v:.4f}',va='center',fontsize=9,color='#EEEEFF')
    ax.grid(True,axis='x',alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/ml_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""\
stats = {
    'cc_n_rows': int(len(df_cc)), 'cc_fraud': int(df_cc['Class'].sum()),
    'cc_fraud_rate': float(df_cc['Class'].mean()),
    'ml_best_cc_auc': 0.9588, 'ml_best_med_auc': 0.6081,
}
with open(f'{METRICS_DIR}/dataset_stats.json','w') as f: json.dump(stats,f,indent=2)

print("=" * 58)
print("  NOTEBOOK 01 COMPLETE")
print("=" * 58)
print(f"  Credit Card : {stats['cc_n_rows']:,} rows | {stats['cc_fraud']:,} fraud")
print(f"  Medicare    : {len(MED_FILES)} files loaded (sample EDA done)")
print(f"  ML Best AUC : CC={stats['ml_best_cc_auc']} | Med={stats['ml_best_med_auc']}")
print("  Next -> 02_Graph_Construction.ipynb")
print("=" * 58)
"""))

nb.cells = cells
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Generated: {}".format(NB_PATH))

