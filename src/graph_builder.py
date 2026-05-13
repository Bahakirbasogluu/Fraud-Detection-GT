"""
graph_builder.py
================
Graph construction utilities for the Fraud Detection GT project.

Credit Card:
  - K-NN similarity graph on PCA-reduced features
  - Node = transaction, Edge = cosine similarity

Medicare (3-year + LEIE):
  - Load 3 separate CMS Medicare Part D CSVs (2017, 2018, 2019)
  - Load LEIE UPDATED.csv (OIG exclusion list)
  - Join on NPI to create fraud labels
  - Build Provider-Drug bipartite graph
  - Project to Provider-Provider graph
"""

import numpy as np
import pandas as pd
import networkx as nx
import pickle
import glob
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Credit Card Graph Builder
# ============================================================

class CreditCardGraphBuilder:
    """
    K-NN similarity graph from the Credit Card Fraud dataset.
    Node = transaction | Edge = cosine similarity in PCA space
    """

    def __init__(self, config: dict):
        cc_cfg = config.get("graph_construction", {}).get("credit_card", {})
        self.k             = cc_cfg.get("knn_k", 10)
        self.method        = cc_cfg.get("method", "knn")
        self.threshold     = cc_cfg.get("threshold", 0.95)
        self.use_pca       = cc_cfg.get("use_pca", True)
        self.pca_components= cc_cfg.get("pca_components", 10)
        self.max_nodes     = cc_cfg.get("max_nodes", 50000)
        self.seed          = config.get("random_seed", 42)

    def load_and_preprocess(self, csv_path: str):
        print(f"[CC] Loading {csv_path} ...")
        df = pd.read_csv(csv_path)

        feature_cols = [c for c in df.columns if c.startswith("V")] + ["Amount"]
        X = df[feature_cols].values
        y = df["Class"].values

        # Stratified subsample to limit memory
        if len(df) > self.max_nodes:
            np.random.seed(self.seed)
            fraud_idx  = np.where(y == 1)[0]
            normal_idx = np.where(y == 0)[0]
            n_normal   = self.max_nodes - len(fraud_idx)
            sample_n   = np.random.choice(normal_idx, n_normal, replace=False)
            idx        = np.concatenate([fraud_idx, sample_n])
            X, y       = X[idx], y[idx]
            print(f"[CC] Subsampled: {len(idx):,} rows ({len(fraud_idx):,} fraud)")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.use_pca:
            pca = PCA(n_components=self.pca_components, random_state=self.seed)
            X_features = pca.fit_transform(X_scaled)
            print(f"[CC] PCA {self.pca_components}D | var explained: "
                  f"{pca.explained_variance_ratio_.sum():.3f}")
        else:
            X_features = X_scaled

        return X_features, y

    def build_knn_graph(self, X: np.ndarray, y: np.ndarray) -> nx.Graph:
        print(f"[CC] Building {self.k}-NN graph on {len(X):,} nodes ...")
        G = nx.Graph()
        for i in range(len(X)):
            G.add_node(i, label=int(y[i]), features=X[i].tolist())

        nbrs = NearestNeighbors(n_neighbors=self.k + 1, metric="cosine", n_jobs=-1)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        for i in range(len(X)):
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                similarity = 1.0 - dist
                G.add_edge(i, int(j), weight=float(similarity))

        print(f"[CC] Graph ready: {G.number_of_nodes():,} nodes | "
              f"{G.number_of_edges():,} edges")
        return G

    def build(self, csv_path: str) -> nx.Graph:
        X, y = self.load_and_preprocess(csv_path)
        return self.build_knn_graph(X, y)


# ============================================================
# Medicare Graph Builder  (3-year CMS + LEIE merge)
# ============================================================

class MedicareGraphBuilder:
    """
    Provider-Drug bipartite graph from CMS Medicare Part D + LEIE data.

    Step 1: Load all Medicare CSVs (2017, 2018, 2019) and concatenate.
    Step 2: Load LEIE UPDATED.csv and extract excluded NPIs.
    Step 3: Join on NPI → fraud_label = 1 if NPI in LEIE exclusions.
    Step 4: Build Provider-Drug bipartite graph.
    Step 5: Project to Provider-Provider weighted graph.
    """

    # Column name candidates for each field (handles CMS naming variations)
    NPI_CANDIDATES    = ["Prscrbr_NPI", "prscrbr_npi", "NPI", "npi",
                         "PRSCRBR_NPI", "provider_npi"]
    DRUG_CANDIDATES   = ["Gnrc_Name", "gnrc_name", "Brnd_Name", "brnd_name",
                         "drug_name", "GNRC_NAME", "BRND_NAME"]
    CLAIMS_CANDIDATES = ["Tot_Clms", "tot_clms", "TOT_CLMS", "total_claims"]
    STATE_CANDIDATES  = ["Prscrbr_State_Abrvtn", "prscrbr_state_abrvtn",
                         "nppes_provider_state", "STATE"]
    COST_CANDIDATES   = ["Tot_Drug_Cst", "tot_drug_cst", "TOT_DRUG_CST"]

    # LEIE column candidates
    LEIE_NPI_CANDIDATES = ["NPI", "npi", "UPIN", "upin"]

    def __init__(self, config: dict):
        med_cfg = config.get("graph_construction", {}).get("medicare", {})
        self.min_drug_count     = med_cfg.get("min_drug_count", 5)
        self.min_provider_count = med_cfg.get("min_provider_count", 10)
        self.geo_edges          = med_cfg.get("geographic_edges", True)
        self.max_providers      = med_cfg.get("max_providers", 30000)
        self.seed               = config.get("random_seed", 42)

    # ── helpers ──────────────────────────────────────────────────────────
    def _find_col(self, columns: list, candidates: list) -> str | None:
        cols_lower = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    # ── Step 1: Load & concatenate Medicare CSVs ─────────────────────────
    def load_medicare(self, med_dir: str) -> pd.DataFrame:
        """
        Load all CSV files from med_dir, add a 'year' column, and concatenate.
        Accepts either a directory path or a list of file paths.
        """
        if isinstance(med_dir, (list, tuple)):
            files = list(med_dir)
        else:
            files = sorted(glob.glob(str(Path(med_dir) / "*.csv")))

        if not files:
            raise FileNotFoundError(
                f"No CSV files found in {med_dir}. "
                "Expected Medicare Part D CSVs for 2017, 2018, 2019."
            )

        dfs = []
        for fp in files:
            year_guess = None
            for yr in ["2017", "2018", "2019"]:
                if yr in str(fp):
                    year_guess = int(yr)
                    break

            print(f"[Medicare] Loading {Path(fp).name} (year={year_guess}) ...")
            df = pd.read_csv(fp, low_memory=False)
            if year_guess:
                df["year"] = year_guess
            dfs.append(df)
            print(f"           {df.shape[0]:,} rows, {df.shape[1]} cols")

        combined = pd.concat(dfs, ignore_index=True)
        print(f"[Medicare] Combined: {combined.shape[0]:,} rows total")
        return combined

    # ── Step 2: Load LEIE exclusion list ─────────────────────────────────
    def load_leie(self, leie_path: str) -> set:
        """
        Load LEIE UPDATED.csv and return a set of excluded NPI strings.
        Also accepts a directory; will auto-detect the CSV inside.
        """
        p = Path(leie_path)
        if p.is_dir():
            candidates = list(p.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError(f"No CSV found in LEIE dir: {leie_path}")
            leie_path = str(candidates[0])
            print(f"[LEIE] Auto-detected: {Path(leie_path).name}")

        print(f"[LEIE] Loading {Path(leie_path).name} ...")
        leie_df = pd.read_csv(leie_path, low_memory=False)
        print(f"[LEIE] Shape: {leie_df.shape} | Columns: {leie_df.columns.tolist()}")

        npi_col = self._find_col(leie_df.columns.tolist(), self.LEIE_NPI_CANDIDATES)
        if npi_col is None:
            print(f"[LEIE] Warning: NPI column not found. Columns: {leie_df.columns.tolist()}")
            return set()

        # NPI values: keep only valid 10-digit numbers
        npis = leie_df[npi_col].astype(str).str.strip()
        # Filter: valid NPI is 10 digits; blank/0 entries are pre-NPI exclusions
        valid_npis = set(n for n in npis if n.isdigit() and len(n) == 10 and n != "0000000000")
        print(f"[LEIE] Valid NPIs (10-digit): {len(valid_npis):,}")
        return valid_npis

    # ── Step 3: Merge LEIE into Medicare → fraud label ───────────────────
    def merge_labels(self, df_med: pd.DataFrame, excluded_npis: set,
                     npi_col: str) -> pd.DataFrame:
        """Add fraud_label column: 1 if provider NPI in LEIE exclusions."""
        df_med = df_med.copy()
        df_med["_npi_str"] = df_med[npi_col].astype(str).str.strip()
        df_med["fraud_label"] = df_med["_npi_str"].isin(excluded_npis).astype(int)
        df_med = df_med.drop(columns=["_npi_str"])

        n_fraud = df_med["fraud_label"].sum()
        print(f"[Merge] Fraud providers (LEIE match): {n_fraud:,} "
              f"({n_fraud/len(df_med)*100:.3f}%)")
        return df_med

    # ── Step 4: Build bipartite graph ────────────────────────────────────
    def build_bipartite(self, df: pd.DataFrame) -> tuple:
        """Build Provider-Drug bipartite graph with fraud labels on providers."""
        cols = df.columns.tolist()
        npi_col    = self._find_col(cols, self.NPI_CANDIDATES)
        drug_col   = self._find_col(cols, self.DRUG_CANDIDATES)
        claims_col = self._find_col(cols, self.CLAIMS_CANDIDATES)
        state_col  = self._find_col(cols, self.STATE_CANDIDATES)
        label_col  = "fraud_label" if "fraud_label" in df.columns else None

        if npi_col is None or drug_col is None:
            raise ValueError(
                f"Could not find NPI or drug columns.\n"
                f"NPI tried: {self.NPI_CANDIDATES}\n"
                f"Drug tried: {self.DRUG_CANDIDATES}\n"
                f"Available: {cols[:20]}"
            )

        print(f"[Bipartite] NPI={npi_col} | Drug={drug_col} "
              f"| Claims={claims_col} | State={state_col}")

        # Clean and filter
        keep = [npi_col, drug_col]
        if claims_col: keep.append(claims_col)
        if state_col:  keep.append(state_col)
        if label_col:  keep.append(label_col)

        df_c = df[keep].dropna(subset=[npi_col, drug_col]).copy()
        df_c[npi_col]  = df_c[npi_col].astype(str).str.strip()
        df_c[drug_col] = df_c[drug_col].astype(str).str.strip().str.upper()

        # Remove providers with too few distinct drugs
        drug_per_prov = df_c.groupby(npi_col)[drug_col].nunique()
        active_prov   = drug_per_prov[drug_per_prov >= self.min_drug_count].index
        df_c = df_c[df_c[npi_col].isin(active_prov)]

        # Remove drugs prescribed by too few providers
        prov_per_drug = df_c.groupby(drug_col)[npi_col].nunique()
        pop_drugs     = prov_per_drug[prov_per_drug >= self.min_provider_count].index
        df_c = df_c[df_c[drug_col].isin(pop_drugs)]

        # Subsample providers if too many
        unique_prov = df_c[npi_col].unique()
        if len(unique_prov) > self.max_providers:
            np.random.seed(self.seed)
            sampled = np.random.choice(unique_prov, self.max_providers, replace=False)
            df_c    = df_c[df_c[npi_col].isin(sampled)]
            unique_prov = df_c[npi_col].unique()

        # Provider fraud labels (max per NPI — LEIE match on any row)
        if label_col:
            prov_labels = (
                df_c.groupby(npi_col)[label_col].max().fillna(0).astype(int)
            )
        else:
            prov_labels = pd.Series(0, index=unique_prov)

        # Provider feature aggregates (for use as node features in GNN)
        prov_features = {}
        if claims_col:
            prov_features["tot_clms"] = df_c.groupby(npi_col)[claims_col].sum()
        if state_col:
            prov_features["state"] = df_c.groupby(npi_col)[state_col].first()

        unique_drugs = df_c[drug_col].unique()

        # Build bipartite graph
        B = nx.Graph()
        for p in unique_prov:
            attrs = {
                "bipartite": 0,
                "node_type": "provider",
                "label": int(prov_labels.get(p, 0)),
            }
            if "tot_clms" in prov_features:
                attrs["tot_clms"] = float(prov_features["tot_clms"].get(p, 0))
            if "state" in prov_features:
                attrs["state"] = str(prov_features["state"].get(p, ""))
            B.add_node(f"P_{p}", **attrs)

        for d in unique_drugs:
            B.add_node(f"D_{d}", bipartite=1, node_type="drug", label=0)

        # Edges
        edges = (
            df_c[[npi_col, drug_col]]
            .drop_duplicates()
            .values
        )
        for prov, drug in edges:
            B.add_edge(f"P_{prov}", f"D_{drug}")

        fraud_cnt = sum(1 for n, d in B.nodes(data=True)
                        if d.get("node_type") == "provider" and d.get("label") == 1)
        print(f"[Bipartite] Providers: {len(unique_prov):,} | "
              f"Drugs: {len(unique_drugs):,} | "
              f"Edges: {B.number_of_edges():,} | "
              f"Fraud providers: {fraud_cnt:,}")
        return B, df_c, npi_col, drug_col

    # ── Step 5: Project bipartite → provider-provider ────────────────────
    def project_to_providers(self, B: nx.Graph) -> nx.Graph:
        from networkx.algorithms import bipartite
        prov_nodes = {n for n, d in B.nodes(data=True) if d.get("bipartite") == 0}
        print(f"[Project] Projecting {len(prov_nodes):,} providers ...")
        G = bipartite.weighted_projected_graph(B, prov_nodes)
        # Copy node attributes from bipartite graph
        for node in G.nodes():
            G.nodes[node].update(B.nodes[node])
        print(f"[Project] Provider graph: {G.number_of_nodes():,} nodes | "
              f"{G.number_of_edges():,} edges")
        return G

    # ── Main entry point ─────────────────────────────────────────────────
    def build(self, med_dir: str, leie_path: str) -> tuple:
        """
        Full pipeline:
          med_dir    → directory with Medicare Part D CSVs (or list of paths)
          leie_path  → path to LEIE UPDATED.csv (or directory containing it)

        Returns: (G_providers, B_bipartite, df_merged, col_info)
        """
        # 1. Load Medicare
        df_med = self.load_medicare(med_dir)

        # 2. Identify NPI column
        npi_col = self._find_col(df_med.columns.tolist(), self.NPI_CANDIDATES)
        if npi_col is None:
            raise ValueError(
                f"NPI column not found in Medicare data.\n"
                f"Tried: {self.NPI_CANDIDATES}\n"
                f"Available: {df_med.columns.tolist()[:20]}"
            )

        # 3. Load LEIE and get excluded NPIs
        try:
            excluded_npis = self.load_leie(leie_path)
        except Exception as e:
            print(f"[LEIE] Warning: Could not load LEIE ({e}). "
                  "Proceeding without fraud labels.")
            excluded_npis = set()

        # 4. Merge labels
        df_med = self.merge_labels(df_med, excluded_npis, npi_col)

        # 5. Build bipartite graph
        B, df_clean, npi_col, drug_col = self.build_bipartite(df_med)

        # 6. Project to provider-provider graph
        G = self.project_to_providers(B)

        col_info = {"npi": npi_col, "drug": drug_col, "label": "fraud_label"}
        return G, B, df_clean, col_info


# ============================================================
# Graph I/O
# ============================================================

def save_graph(G: nx.Graph, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"[IO] Saved: {path}")


def load_graph(path: str) -> nx.Graph:
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_summary(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    labels  = [d.get("label", -1) for _, d in G.nodes(data=True)]
    n_fraud = sum(1 for l in labels if l == 1)
    n_norm  = sum(1 for l in labels if l == 0)
    return {
        "n_nodes":      n,
        "n_edges":      e,
        "n_fraud":      n_fraud,
        "n_normal":     n_norm,
        "fraud_rate":   n_fraud / n if n > 0 else 0,
        "density":      nx.density(G),
        "avg_degree":   2 * e / n if n > 0 else 0,
        "is_connected": nx.is_connected(G),
    }
