"""
colab_setup.py
==============
Shared helper for Google Colab + Drive setup.
Embedded into every notebook as the first cell.

Drive structure assumed:
  MyDrive/fraudDataset/
    creditcard/creditcard.csv
    leie/UPDATED.csv
    medicare/Medicare_Part_D_...2017.csv
    medicare/Medicare_Part_D_...2018.csv
    medicare/Medicare_Part_D_...2019.csv
    Fraud-Detection-GT/  (this project)
"""

# ── Colab Drive mount cell ─────────────────────────────────────────────────────
DRIVE_MOUNT_CODE = '''\
# ============================================================
# Google Colab Setup -- Run this cell FIRST every session
# ============================================================
import os, sys, glob

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ── Roots ────────────────────────────────────────────────────
DRIVE_ROOT   = "/content/drive/MyDrive/fraudDataset"
PROJECT_ROOT = f"{DRIVE_ROOT}/Fraud-Detection-GT"

# Add project src/ to Python path so we can import from src/
sys.path.insert(0, PROJECT_ROOT)

# ── Dataset Paths ─────────────────────────────────────────────
CC_PATH  = f"{DRIVE_ROOT}/creditcard/creditcard.csv"
LEIE_PATH = f"{DRIVE_ROOT}/leie/UPDATED.csv"

# Medicare: 3 separate yearly files (auto-detect names)
MED_DIR = f"{DRIVE_ROOT}/medicare"
MED_FILES = sorted(glob.glob(f"{MED_DIR}/*.csv"))  # finds all CSVs

# ── Output Paths (on Drive  →  survive session restarts) ─────
OUTPUTS_DIR = f"{PROJECT_ROOT}/outputs"
FIGURES_DIR = f"{PROJECT_ROOT}/outputs/figures"
METRICS_DIR = f"{PROJECT_ROOT}/outputs/metrics"
MODELS_DIR  = f"{PROJECT_ROOT}/outputs/models"
GRAPHS_DIR  = f"{PROJECT_ROOT}/data/graphs"

# ML baseline pickles (optional -- upload Fraud-Detection project too)
ML_PICKLES_DIR = f"{DRIVE_ROOT}/Fraud-Detection/pickled_storage"

# Create output directories (idempotent)
for _d in [OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, GRAPHS_DIR,
           f"{GRAPHS_DIR}/credit_card", f"{GRAPHS_DIR}/medicare"]:
    os.makedirs(_d, exist_ok=True)

# Load config.yaml
import yaml
_cfg_path = f"{PROJECT_ROOT}/config.yaml"
CONFIG = {}
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        CONFIG = yaml.safe_load(_f)

# ── Status Report ─────────────────────────────────────────────
print("=" * 58)
print("  Fraud Detection GT  |  Colab + Drive Environment")
print("=" * 58)
print(f"  Project Root  : {PROJECT_ROOT}")
print(f"  Credit Card   : {CC_PATH}")
print(f"  LEIE file     : {LEIE_PATH}")
print(f"  Medicare files: {len(MED_FILES)} found in {MED_DIR}")
for _f in MED_FILES:
    print(f"    - {os.path.basename(_f)}")
print()
print(f"  Outputs       : {OUTPUTS_DIR}")
print()

# Verify
_ok = True
for _path, _label in [(CC_PATH, "creditcard/creditcard.csv"),
                       (LEIE_PATH, "leie/UPDATED.csv")]:
    if os.path.exists(_path):
        _mb = os.path.getsize(_path)/1e6
        print(f"  [OK]  {_label} ({_mb:.0f} MB)")
    else:
        print(f"  [!!]  {_label} NOT FOUND")
        _ok = False

if len(MED_FILES) == 0:
    print("  [!!]  No Medicare CSVs found in medicare/")
    _ok = False
elif len(MED_FILES) < 3:
    print(f"  [??]  Only {len(MED_FILES)} Medicare file(s) found (expected 3)")
else:
    total_mb = sum(os.path.getsize(f)/1e6 for f in MED_FILES)
    print(f"  [OK]  medicare/ ({len(MED_FILES)} files, {total_mb:.0f} MB total)")

print()
print("  Ready!" if _ok else "  WARNING: Some files missing -- check Drive paths above")
print("=" * 58)
'''

# ── Package installation cell ──────────────────────────────────────────────────
INSTALL_CODE = '''\
# ============================================================
# Install required packages (once per Colab session)
# ============================================================
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(pkgs))

print("Installing packages...")
_pip("networkx", "python-louvain", "pyvis", "pyyaml", "plotly", "tqdm")

# PyTorch Geometric -- auto-detect CUDA
import torch
_cuda = torch.version.cuda
if _cuda:
    _cu  = "cu" + _cuda.replace(".", "")
    _tv  = torch.__version__.split("+")[0]
    _url = f"https://data.pyg.org/whl/torch-{_tv}+{_cu}.html"
    print(f"CUDA {_cuda} detected -- installing PyG GPU build")
    _pip("torch-geometric")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "torch-scatter", "torch-sparse", "-f", _url])
else:
    print("No CUDA -- installing PyG CPU build")
    _pip("torch-geometric")

print("All packages installed successfully!")
'''
