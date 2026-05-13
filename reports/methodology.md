# Methodology: Graph Theory-Based Fraud Detection

## Overview

This document describes the graph theory methodology used in this project,
and why each technique is appropriate for fraud detection.

---

## 1. Problem Formulation

### Tabular Formulation (ML Baseline)
Given feature matrix $X \in \mathbb{R}^{n \times d}$ and labels $y \in \{0, 1\}^n$,
find a decision function $f: \mathbb{R}^d \to [0,1]$ that scores anomalies.

**Limitation:** Ignores relationships between entities.

### Graph Formulation (This Project)
Given a graph $G = (V, E, X, Y)$ where:
- $V$ = nodes (transactions or providers)
- $E$ = edges (similarity or co-prescription relationships)
- $X \in \mathbb{R}^{|V| \times d}$ = node feature matrix
- $Y \in \{0, 1\}^{|V|}$ = fraud labels

Find a scoring function $f: G \to [0,1]^{|V|}$ that estimates fraud probability
using **both** node features AND graph structure.

---

## 2. Graph Construction

### Credit Card: K-NN Similarity Graph
- Node = transaction
- Edge $(i, j)$: Transaction $i$ and $j$ are cosine-similar in PCA space
- $A_{ij} = 1$ if $\cos(x_i, x_j) \geq \tau$ or $j \in \text{k-NN}(i)$

**Rationale:** Fraudulent transactions form tight clusters in feature space.
The K-NN graph makes this cluster structure explicit.

### Medicare: Provider-Drug Bipartite Graph
- $V = V_P \cup V_D$ where $V_P$ = providers, $V_D$ = drugs
- Edge $(p, d)$: Provider $p$ prescribed drug $d$
- Project onto $V_P$: $A_{p1,p2} = |D(p1) \cap D(p2)|$ (shared drugs)

**Rationale:** Medicare fraud involves collusion — fraud providers share
unusual drug prescribing patterns. The bipartite projection makes this
structure explicit and quantifiable.

---

## 3. Classical Graph Analysis

### 3.1 Centrality Measures

**PageRank** (Brin & Page, 1998):
$$PR(v) = \frac{1-\alpha}{|V|} + \alpha \sum_{u \in \mathcal{N}(v)} \frac{PR(u)}{k(u)}$$

Converges to the leading eigenvector of the transition matrix.
Fraud nodes embedded in fraud rings tend to accumulate high PageRank.

**Betweenness Centrality**:
$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma(s,t|v)}{\sigma(s,t)}$$

Measures brokerage. Fraud coordinators in rings act as brokers.

**Clustering Coefficient**:
$$C(v) = \frac{2|\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in E\}|}{k(v)(k(v)-1)}$$

Low clustering = radiating star pattern (ghost billing hub).

### 3.2 Community Detection

**Louvain Algorithm** (Blondel et al., 2008):
Greedily optimizes modularity:
$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

Fraud rings form high-modularity communities.

---

## 4. Spectral Graph Theory

### 4.1 Graph Laplacian
$$L = D - A, \quad \mathcal{L} = D^{-1/2} L D^{-1/2}$$

Positive semi-definite. Eigenvalues $0 = \lambda_1 \leq \cdots \leq \lambda_n$.

### 4.2 Algebraic Connectivity (Fiedler Value, 1973)
$\lambda_2$ = second smallest eigenvalue of $L$.
- Small $\lambda_2$ → easy to cut graph into 2 components
- Fiedler vector = eigenvector of $\lambda_2$ → optimal bisection

**Fraud interpretation:** If fraud nodes partition with the Fiedler vector,
they form structurally separable communities.

### 4.3 Chebyshev Spectral Convolution → GCN

Spectral convolution of feature signal $x$ with filter $g_\theta$:
$$g_\theta * x = U g_\theta(\Lambda) U^T x$$

Approximating with degree-K Chebyshev polynomials (Hammond et al., 2011):
$$g_\theta * x \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x$$

Setting K=1 and $\lambda_{max}=2$: (Kipf & Welling, 2017)
$$H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})$$

This is the **GCN layer** — spectral convolution made efficient.

---

## 5. Graph Neural Networks

### 5.1 Message Passing Framework (Gilmer et al., 2017)

All GNNs follow:
$$m_v^{(k)} = \text{AGGREGATE}^{(k)}\left(\{ h_u^{(k-1)} : u \in \mathcal{N}(v) \}\right)$$
$$h_v^{(k)} = \text{UPDATE}^{(k)}\left( h_v^{(k-1)}, m_v^{(k)} \right)$$

### 5.2 Models

| Model | AGGREGATE | UPDATE | Reference |
|-------|-----------|--------|-----------|
| GCN | $\hat{A} H$ | $\sigma(W \cdot)$ | Kipf & Welling 2017 |
| GraphSAGE | Mean/Max pool | $W[h_v \| m_v]$ | Hamilton et al. 2017 |
| GAT | $\sum_u \alpha_{uv} h_u$ | $\sigma(W \cdot)$ | Veličković et al. 2018 |

GAT attention coefficient:
$$\alpha_{ij} = \text{softmax}_j\left( \text{LeakyReLU}(a^T [Wh_i \| Wh_j]) \right)$$

### 5.3 DOMINANT (Ding et al., 2019)

Anomaly score = weighted sum of two reconstruction errors:
$$s(v) = \alpha \cdot \left\| a_v - \hat{a}_v \right\|^2 + (1-\alpha) \cdot \left\| x_v - \hat{x}_v \right\|^2$$

where $a_v$ is row $v$ of adjacency matrix and $x_v$ is node feature vector.

---

## 6. Evaluation Protocol

### Metrics
- **AUC-ROC**: Threshold-invariant, standard for imbalanced data
- **AUC-PRC (Average Precision)**: Critical when fraud << normal
- **F1 @ optimal threshold**: Practical detection performance

### Handling Class Imbalance
- Supervised GNNs: `pos_weight = n_normal / n_fraud` in BCELoss
- Classical: `class_weight='balanced'` in RandomForest
- Unsupervised (DOMINANT): No labels required

### Comparison Protocol
- Same train/val/test splits (70/10/20, stratified)
- Same datasets and preprocessing
- 5-fold cross-validation for classical methods
- Single run with fixed seed for GNNs

---

## References

1. Kipf & Welling (2017). Semi-Supervised Classification with GCN. ICLR.
2. Hamilton et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
3. Veličković et al. (2018). Graph Attention Networks. ICLR.
4. Ding et al. (2019). Deep Anomaly Detection on Attributed Networks. SDM.
5. Blondel et al. (2008). Fast unfolding of communities in large networks. J. Stat. Mech.
6. Fiedler (1973). Algebraic connectivity of graphs. Czech. Math. J.
7. Brin & Page (1998). The anatomy of a large-scale hypertextual web search engine.
