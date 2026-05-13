"""
graph_features.py
=================
Classical graph theory features for fraud detection:
  - Centrality measures (PageRank, betweenness, closeness, eigenvector, degree)
  - Community detection (Louvain, Girvan-Newman)
  - Spectral features (Laplacian eigenvalues, Fiedler vector)
  - Clustering coefficients
  - Motif analysis
"""

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
warnings.filterwarnings("ignore")
from typing import Optional


# ============================================================
# Centrality Features
# ============================================================

def compute_centrality_features(G: nx.Graph, verbose: bool = True) -> pd.DataFrame:
    """
    Compute multiple centrality measures for all nodes.
    
    Returns DataFrame with columns:
      node_id, degree, degree_centrality, pagerank, betweenness_centrality,
      closeness_centrality, eigenvector_centrality, clustering_coeff, label
    """
    if verbose:
        print(f"[Centrality] Computing features for {G.number_of_nodes()} nodes...")

    records = []

    # 1. Degree
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)

    # 2. PageRank
    if verbose: print("  → PageRank...")
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-6)

    # 3. Betweenness Centrality (expensive — approximate for large graphs)
    n = G.number_of_nodes()
    if verbose: print(f"  → Betweenness (k=min({n},500) approx)...")
    k_approx = min(n, 500)
    betweenness = nx.betweenness_centrality(G, k=k_approx, normalized=True, seed=42)

    # 4. Closeness Centrality
    if verbose: print("  → Closeness...")
    closeness = nx.closeness_centrality(G)

    # 5. Eigenvector Centrality
    if verbose: print("  → Eigenvector centrality...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {n: 0.0 for n in G.nodes()}

    # 6. Clustering Coefficient
    if verbose: print("  → Clustering coefficient...")
    clustering = nx.clustering(G)

    # Assemble
    for node in G.nodes():
        node_data = G.nodes[node]
        records.append({
            "node_id": node,
            "degree": degree_dict[node],
            "degree_centrality": degree_centrality[node],
            "pagerank": pagerank[node],
            "betweenness_centrality": betweenness[node],
            "closeness_centrality": closeness[node],
            "eigenvector_centrality": eigenvector[node],
            "clustering_coeff": clustering[node],
            "label": node_data.get("label", -1),
        })

    df = pd.DataFrame(records)
    if verbose:
        fraud_mask = df["label"] == 1
        print(f"\n[Centrality] Summary (Fraud vs Normal):")
        for col in ["degree", "pagerank", "betweenness_centrality", "clustering_coeff"]:
            fraud_mean = df[fraud_mask][col].mean()
            normal_mean = df[~fraud_mask & (df["label"] == 0)][col].mean()
            print(f"  {col}: fraud={fraud_mean:.4f}, normal={normal_mean:.4f}")

    return df


# ============================================================
# Community Detection
# ============================================================

def detect_communities_louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42) -> dict:
    """Louvain community detection. Returns node → community_id mapping."""
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=seed)
        n_communities = len(set(partition.values()))
        print(f"[Community] Louvain: {n_communities} communities detected")
        return partition
    except ImportError:
        print("[Community] python-louvain not installed. Using greedy modularity.")
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G, resolution=resolution, seed=seed)
        partition = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                partition[node] = cid
        n_communities = len(communities)
        print(f"[Community] Greedy modularity: {n_communities} communities")
        return partition


def detect_communities_label_propagation(G: nx.Graph, seed: int = 42) -> dict:
    """Label propagation community detection."""
    from networkx.algorithms.community import label_propagation_communities
    communities = label_propagation_communities(G, seed=seed)
    partition = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    print(f"[Community] Label Propagation: {len(set(partition.values()))} communities")
    return partition


def community_fraud_analysis(G: nx.Graph, partition: dict) -> pd.DataFrame:
    """
    Analyze fraud concentration per community.
    Returns DataFrame: community_id, size, fraud_count, fraud_rate, modularity_contribution
    """
    records = []
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    for cid, nodes in communities.items():
        labels = [G.nodes[n].get("label", 0) for n in nodes]
        fraud_count = sum(labels)
        records.append({
            "community_id": cid,
            "size": len(nodes),
            "fraud_count": fraud_count,
            "fraud_rate": fraud_count / len(nodes) if nodes else 0,
        })

    df = pd.DataFrame(records).sort_values("fraud_rate", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# Spectral Graph Analysis
# ============================================================

def compute_laplacian(G: nx.Graph, normalized: bool = True) -> sp.csr_matrix:
    """Compute (normalized) graph Laplacian as sparse matrix."""
    if normalized:
        L = nx.normalized_laplacian_matrix(G).astype(float)
    else:
        L = nx.laplacian_matrix(G).astype(float)
    return L


def compute_spectral_features(G: nx.Graph, n_eigenvectors: int = 20) -> dict:
    """
    Compute spectral features of the graph Laplacian:
      - k smallest eigenvalues
      - Fiedler value (algebraic connectivity) = 2nd smallest eigenvalue
      - Fiedler vector (bottleneck structure)
      - Spectral gap
    """
    print(f"[Spectral] Computing {n_eigenvectors} eigenvectors for {G.number_of_nodes()} nodes...")
    L = compute_laplacian(G, normalized=True)

    k = min(n_eigenvectors, G.number_of_nodes() - 1)
    eigenvalues, eigenvectors = spla.eigsh(L, k=k, which="SM")

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    fiedler_value = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    fiedler_vector = eigenvectors[:, 1] if eigenvectors.shape[1] > 1 else None
    spectral_gap = eigenvalues[2] - eigenvalues[1] if len(eigenvalues) > 2 else 0.0

    print(f"[Spectral] Fiedler value (algebraic connectivity): {fiedler_value:.6f}")
    print(f"[Spectral] Spectral gap: {spectral_gap:.6f}")

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "fiedler_value": fiedler_value,
        "fiedler_vector": fiedler_vector,
        "spectral_gap": spectral_gap,
        "node_order": list(G.nodes()),
    }


def spectral_clustering_labels(G: nx.Graph, n_clusters: int = 10, seed: int = 42) -> np.ndarray:
    """
    Spectral clustering using k smallest eigenvectors of the Laplacian.
    Returns cluster label array aligned with G.nodes() order.
    """
    from sklearn.cluster import KMeans

    spec = compute_spectral_features(G, n_eigenvectors=n_clusters + 1)
    # Use eigenvectors 1..n_clusters (skip the constant eigenvector 0)
    embedding = spec["eigenvectors"][:, 1:n_clusters + 1]

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(embedding)
    return cluster_labels, spec


# ============================================================
# Motif Analysis
# ============================================================

def count_triangles(G: nx.Graph) -> dict:
    """Count triangles per node."""
    triangles = nx.triangles(G)
    return triangles


def motif_analysis(G: nx.Graph) -> dict:
    """
    Analyze graph motifs relevant to fraud detection.
    Returns statistics on triangles, stars, and path lengths.
    """
    triangles = count_triangles(G)
    total_triangles = sum(triangles.values()) // 3

    # Star analysis: nodes with degree >> average
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    std_degree = np.std(degrees)
    hub_threshold = avg_degree + 2 * std_degree
    hub_nodes = [n for n, d in G.degree() if d > hub_threshold]

    # Fraud in hubs
    fraud_hubs = [n for n in hub_nodes if G.nodes[n].get("label") == 1]

    print(f"[Motif] Total triangles: {total_triangles:,}")
    print(f"[Motif] Hub nodes (degree > avg+2σ): {len(hub_nodes)} ({len(fraud_hubs)} fraud)")

    return {
        "total_triangles": total_triangles,
        "triangle_per_node": triangles,
        "hub_nodes": hub_nodes,
        "fraud_hub_nodes": fraud_hubs,
        "hub_threshold": hub_threshold,
        "avg_degree": avg_degree,
        "std_degree": std_degree,
    }
