"""
generate_notebooks.py
=====================
Master script to generate all 6 project notebooks from Python scripts.

Usage:
    cd Fraud-Detection-GT/
    python scripts/generate_notebooks.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "nb_01_eda_baseline.py",
    "nb_02_graph_construction.py",
    "nb_03_classical_graph_analysis.py",
    "nb_04_spectral_methods.py",
    "nb_05_gnn_models.py",
    "nb_06_comparison_results.py",
]

SCRIPTS_DIR    = Path(__file__).parent
PROJECT_ROOT   = SCRIPTS_DIR.parent
NOTEBOOKS_DIR  = PROJECT_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("  Fraud Detection GT -- Notebook Generator")
    print("=" * 60)
    print("  Scripts dir : {}".format(SCRIPTS_DIR))
    print("  Output dir  : {}".format(NOTEBOOKS_DIR))
    print()

    success, failed = [], []

    for script in SCRIPTS:
        script_path = SCRIPTS_DIR / script
        if not script_path.exists():
            print("  [MISSING] {}".format(script))
            failed.append(script)
            continue

        print("  [RUN] {} ...".format(script))
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True,
            cwd=str(SCRIPTS_DIR),
        )

        if result.returncode == 0:
            print("  [OK]  {}".format(script))
            success.append(script)
        else:
            print("  [ERR] {}".format(script))
            print(result.stderr[-600:])
            failed.append(script)

    print()
    print("  Generated: {}/{} notebooks".format(len(success), len(SCRIPTS)))
    if failed:
        print("  Failed   : {}".format(failed))

    print()
    print("  Generated notebooks:")
    for nb in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        size_kb = nb.stat().st_size / 1024
        print("    [NB] {}  ({:.1f} KB)".format(nb.name, size_kb))

    print()
    print("Done! Upload notebooks/ and src/ to Google Drive.")
    print("Colab execution order: 01 -> 02 -> 03 -> 04 -> 05 -> 06")


if __name__ == "__main__":
    main()
