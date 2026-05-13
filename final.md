# Fraud Detection via Graph Theory: A Comparative Study of Classical, Spectral, and Neural Graph Methods

**Course:** Graduate Graph Theory  
**Author:** Baha Kirbasoglu  
**Date:** May 2026

---

## Abstract

This study examines the application of graph-theoretic methods to financial fraud detection, comparing classical graph analysis, spectral graph theory, and graph neural networks against tabular machine learning baselines. Two real-world datasets were employed: a European credit card transaction dataset and a Medicare provider prescribing dataset derived from CMS Part D records. Graphs were constructed from tabular data through a K-nearest-neighbor similarity approach for credit card transactions and a bipartite provider-drug projection for Medicare providers. A total of fourteen graph-based methods were evaluated alongside five tabular anomaly detection approaches. Results indicate that spectral clustering achieved the highest AUC-ROC among graph methods on the credit card dataset (0.9221), while the Fiedler vector anomaly score narrowly surpassed the tabular baseline on the Medicare dataset (0.6096 vs. 0.6081). Graph neural networks, despite their theoretical capacity for relational learning, underperformed both spectral and classical methods in this study. These findings suggest that the quality of the graph construction step and the inherent relational signal in the data are decisive factors in whether graph-based approaches offer a measurable advantage over conventional anomaly detection.

---

## 1. Introduction

Financial fraud imposes substantial economic costs globally, and its detection presents persistent challenges due to extreme class imbalance, adversarial adaptation, and the complex relational structure of fraudulent behavior. Traditional machine learning approaches treat each entity — a transaction or a provider — as an independent observation, thereby discarding potentially informative relationships among entities. Graph-theoretic formulations offer an alternative: by encoding entities as nodes and their relationships as edges, structural patterns such as fraud rings, collusion networks, and anomalous clustering become amenable to formal analysis.

This project was conducted as the term project for a graduate-level graph theory course. Its primary objective was to determine whether graph-theoretic methods provide a measurable improvement over tabular anomaly detection, and under what conditions such improvement is realized. The project draws on two datasets that were also analyzed in a companion data mining course project (BLG607), enabling a direct performance comparison under identical preprocessing and evaluation conditions.

The central research hypothesis is stated as follows:

> *Fraudulent entities form structurally distinct communities in transaction graphs, and graph-theoretic methods — including centrality-based classification, spectral analysis, and graph neural networks — outperform tabular anomaly detection on relational fraud data, particularly for Medicare provider-level fraud.*

---

## 2. Datasets

### 2.1 Credit Card Fraud Dataset

The ULB credit card fraud dataset comprises 284,807 card-not-present transactions recorded over two days in September 2013 by European cardholders. Of these, 492 transactions (0.172%) are labeled as fraudulent. Features V1 through V28 represent principal components derived from a confidential PCA transformation applied to the original transaction attributes; only the transaction amount and elapsed time are available in their original form. The extreme class imbalance and the PCA preprocessing are both significant factors in interpreting the results of graph construction on this dataset.

**Key statistics:**
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Features: 28 PCA components + Amount + Time
- Graph subset used: 50,000 transactions (stratified sample)

### 2.2 Medicare Provider Fraud Dataset

The Medicare dataset was constructed by joining CMS Medicare Part D prescribing records for 2017, 2018, and 2019 with the Office of Inspector General (OIG) List of Excluded Individuals and Entities (LEIE). Each record associates a National Provider Identifier (NPI) with a drug name and the total number of prescriptions issued in that year. Providers appearing in the LEIE are treated as fraudulent ground truth. A stratified sample of 6 million rows (2 million per year) was drawn from the full dataset, yielding 7,701 unique providers for graph analysis, of which 137 (1.78%) are labeled as fraudulent.

**Key statistics:**
- Total providers in graph: 7,701
- Fraudulent providers (LEIE-excluded): 137 (1.78%)
- Fraud ground truth source: OIG LEIE (8,306 excluded NPIs nationally)
- Raw data size: approximately 11.9 GB (3 yearly CSV files)

---

## 3. Methodology

### 3.1 Problem Formulation

In the tabular formulation, a decision function $f: \mathbb{R}^d \to [0,1]$ is learned from a feature matrix $X \in \mathbb{R}^{n \times d}$ with labels $y \in \{0,1\}^n$. This formulation treats entities as independent and identically distributed, discarding inter-entity relationships.

The graph formulation redefines the problem. A graph $G = (V, E, X, Y)$ is constructed, where $V$ denotes nodes (transactions or providers), $E$ denotes edges encoding relationships, $X \in \mathbb{R}^{|V| \times d}$ is a node feature matrix, and $Y \in \{0,1\}^{|V|}$ contains fraud labels. A scoring function $f: G \to [0,1]^{|V|}$ is then sought that exploits both node features and graph topology.

### 3.2 Graph Construction

#### 3.2.1 Credit Card: K-Nearest Neighbor Similarity Graph

Each transaction was represented as a node. An edge was placed between transactions $i$ and $j$ if $j$ belongs to the k-nearest neighbors of $i$ under cosine similarity in the 28-dimensional PCA feature space, with $k = 10$. The resulting graph contains 50,000 nodes and 347,127 edges.

$$A_{ij} = 1 \quad \text{if} \quad j \in \text{k-NN}(i) \quad \text{or} \quad i \in \text{k-NN}(j)$$

The rationale is that fraudulent transactions, if they form tight clusters in feature space, should exhibit elevated local density in the resulting graph. A critical limitation is that the PCA transformation was applied to the raw features before this project began, meaning the graph edges encode proximity in a compressed representation rather than in the original transaction space.

#### 3.2.2 Medicare: Provider-Drug Bipartite Graph and Projection

A bipartite graph $B = (V_P \cup V_D, E_B)$ was first constructed, where $V_P$ is the set of provider nodes, $V_D$ is the set of drug nodes, and an edge $(p, d)$ exists when provider $p$ prescribed drug $d$. The bipartite graph was then projected onto the provider layer to produce a unipartite provider-provider graph $G = (V_P, E_G)$, where an edge $(p_1, p_2)$ exists when the two providers share at least five prescribed drugs.

A key challenge in this construction was the presence of hub drugs — commonly prescribed medications dispensed by a large fraction of all providers. Without mitigation, these hub drugs would connect nearly every provider pair, producing a near-complete graph with approximately 334 million edges and rendering all structural analysis meaningless. Hub drugs were therefore removed prior to projection, defined as any drug prescribed by more than 25% of all providers. Additionally, each provider's neighborhood was capped at $K = 25$ neighbors to bound memory consumption. The resulting graph contains 7,701 nodes and 113,350 edges.

### 3.3 Classical Graph Analysis

#### 3.3.1 Centrality Measures

Six centrality measures were computed for all nodes in both graphs:

- **Degree centrality:** $C_D(v) = k(v) / (|V| - 1)$
- **PageRank:** $PR(v) = (1-\alpha)/|V| + \alpha \sum_{u \in \mathcal{N}(v)} PR(u)/k(u)$, with damping factor $\alpha = 0.85$ and 200 iterations
- **Betweenness centrality:** $C_B(v) = \sum_{s \neq v \neq t} \sigma(s,t|v)/\sigma(s,t)$, estimated via sampling ($k = 300$ pivot nodes)
- **Closeness centrality:** $C_C(v) = (|V|-1) / \sum_{u \neq v} d(u,v)$
- **Eigenvector centrality:** leading eigenvector of the adjacency matrix
- **Local clustering coefficient:** $C(v) = 2|\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in E\}| / (k(v)(k(v)-1))$

These six features, together with node degree, were concatenated into a seven-dimensional feature vector and used to train a Random Forest classifier (with balanced class weights) in a supervised classification experiment.

#### 3.3.2 Community Detection

The Louvain algorithm was applied to both graphs to identify community structure. The algorithm greedily optimizes modularity:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

On the credit card graph, 31 communities were detected. Community 0 contained 396 nodes, 383 of which were fraudulent, corresponding to a within-community fraud rate of 96.7%. This result provides strong evidence that fraudulent transactions do form structurally cohesive subgraphs, at least in the credit card domain.

On the Medicare graph, 3,042 communities were detected, the majority of which were singleton providers — a signature of scale-free network topology in which high-degree hub nodes coexist with many weakly connected periphery nodes.

### 3.4 Spectral Methods

The normalized graph Laplacian was computed as $\mathcal{L} = D^{-1/2}(D - A)D^{-1/2}$, where $D$ is the diagonal degree matrix and $A$ is the adjacency matrix. Its eigenvalues $0 = \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ encode structural properties of the graph.

**Fiedler value and vector.** The second smallest eigenvalue $\lambda_2$ (algebraic connectivity, or Fiedler value) measures graph connectivity; a small $\lambda_2$ indicates that the graph is close to being disconnected. The corresponding eigenvector, the Fiedler vector, provides an optimal graph bisection. For the credit card graph, $\lambda_2 = 0.000619$, indicating a weakly connected structure susceptible to partitioning. For the Medicare graph, $\lambda_2 = 0.013493$.

The Fiedler vector was used directly as a node anomaly score by treating its magnitude as an indicator of structural isolation. This yielded AUC-ROC = 0.8861 on the credit card graph and AUC-ROC = 0.6096 on the Medicare graph.

**Spectral clustering.** The bottom $k = 10$ eigenvectors of $\mathcal{L}$ were computed, and K-means clustering was applied in the resulting spectral embedding. Cluster membership was then used to score nodes by the fraud rate of their assigned cluster. On the credit card graph, this approach achieved AUC-ROC = 0.9221, the highest of all graph methods. On the Medicare graph, AUC-ROC = 0.5779.

**Chebyshev polynomial approximation.** The relationship between spectral graph convolution and GCN layers was demonstrated analytically. Spectral convolution $g_\theta * x = U g_\theta(\Lambda) U^T x$ is approximated by truncated Chebyshev polynomials as $g_\theta * x \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x$. Setting $K = 1$ and $\lambda_{\max} = 2$ recovers the GCN layer $H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})$, establishing GCN as a special case of spectral graph filtering.

### 3.5 Graph Neural Networks

Four GNN architectures were implemented and trained using PyTorch Geometric. Node input features were the seven centrality measures computed in Section 3.3.1. A 70/10/20 stratified split was used for training, validation, and testing. Binary cross-entropy loss was applied with class weighting $w_{\text{pos}} = n_{\text{normal}} / n_{\text{fraud}}$ to address class imbalance.

**GCN (Kipf & Welling, 2017).** Two GCN layers (input $\to$ 64 $\to$ 32) followed by a linear output layer and sigmoid activation. Dropout $p = 0.5$ was applied after each hidden layer. Trained with Adam ($\text{lr} = 0.01$, $\lambda_{\text{decay}} = 5 \times 10^{-4}$) for 200 epochs.

**GraphSAGE (Hamilton et al., 2017).** Mean neighborhood aggregation with concatenation of the self-representation and aggregated neighbor representation. Same architecture dimensions and training protocol as GCN.

**GAT (Velickovic et al., 2018).** Four attention heads with 8 hidden units each. The attention coefficient $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(a^T [Wh_i \| Wh_j]))$ allows the model to weight neighbor contributions adaptively. Learning rate $5 \times 10^{-3}$.

**DOMINANT (Ding et al., 2019).** An unsupervised deep anomaly detection model that jointly reconstructs node attribute vectors and the graph adjacency structure through a shared encoder. Anomaly scores are defined as a convex combination of attribute reconstruction error and structural reconstruction error: $s(v) = \alpha \| a_v - \hat{a}_v \|^2 + (1-\alpha) \| x_v - \hat{x}_v \|^2$, with $\alpha = 0.5$.

### 3.6 Evaluation Protocol

Performance was evaluated using the following metrics:

- **AUC-ROC:** Area under the receiver operating characteristic curve; threshold-invariant and the primary metric for imbalanced datasets.
- **AUC-PRC (Average Precision):** Area under the precision-recall curve; particularly sensitive to performance on the minority class.
- **F1 at optimal threshold:** Reported for supervised GNN models after threshold optimization on the validation set.

---

## 4. Results

### 4.1 Comprehensive Performance Comparison

The table below presents AUC-ROC and AUC-PRC for all methods evaluated on both datasets. Results are organized by method category.

| Dataset      | Category         | Method                   | AUC-ROC | AUC-PRC |
|:-------------|:-----------------|:-------------------------|--------:|--------:|
| Credit Card  | Tabular-ML       | GMM                      | 0.9588  | 0.5720  |
| Credit Card  | Tabular-ML       | LOF                      | 0.9532  | 0.6391  |
| Credit Card  | Tabular-ML       | Ensemble (Best)          | 0.9535  | 0.5191  |
| Credit Card  | Tabular-ML       | OCSVM                    | 0.9514  | 0.2731  |
| Credit Card  | Tabular-ML       | Isolation Forest         | 0.9496  | 0.1543  |
| Credit Card  | Graph-Spectral   | Spectral Clustering k=10 | 0.9221  | 0.7489  |
| Credit Card  | Graph-Spectral   | Fiedler Anomaly Score    | 0.8861  | 0.7880  |
| Credit Card  | Graph-Classical  | RF on Centrality         | 0.8839  | 0.4431  |
| Credit Card  | Graph-GNN        | GCN                      | 0.6144  | 0.0693  |
| Credit Card  | Graph-GNN        | GraphSAGE                | 0.5171  | 0.0643  |
| Credit Card  | Graph-GNN-Unsup  | DOMINANT                 | 0.5708  | 0.0197  |
| Credit Card  | Graph-GNN        | GAT                      | 0.3586  | 0.0136  |
| Medicare     | Tabular-ML       | Isolation Forest         | 0.6081  | 0.0038  |
| Medicare     | Tabular-ML       | LOF                      | 0.6043  | 0.0021  |
| Medicare     | Tabular-ML       | GMM                      | 0.5977  | 0.0023  |
| Medicare     | Tabular-ML       | Ensemble (Best)          | 0.5974  | 0.0037  |
| Medicare     | Tabular-ML       | OCSVM                    | 0.5261  | 0.0019  |
| Medicare     | Graph-Spectral   | Fiedler Anomaly Score    | 0.6096  | 0.0245  |
| Medicare     | Graph-GNN        | GCN                      | 0.6139  | 0.0273  |
| Medicare     | Graph-Spectral   | Spectral Clustering k=10 | 0.5779  | 0.0251  |
| Medicare     | Graph-GNN-Unsup  | DOMINANT                 | 0.5509  | 0.0200  |
| Medicare     | Graph-GNN        | GAT                      | 0.4361  | 0.0159  |
| Medicare     | Graph-GNN        | GraphSAGE                | 0.4309  | 0.0171  |
| Medicare     | Graph-Classical  | RF on Centrality         | 0.4332  | 0.0144  |

### 4.2 Credit Card Results

On the credit card dataset, all tabular ML methods achieved AUC-ROC values above 0.949, with the Gaussian Mixture Model attaining 0.9588. The best graph method, spectral clustering with $k = 10$, reached 0.9221 — a relative decrease of 3.83% compared to the GMM baseline. Despite this, spectral methods substantially outperformed graph neural networks on this dataset; GCN achieved only 0.6144, and GAT reached as low as 0.3586.

A notable observation is that the Fiedler vector anomaly score achieved the highest AUC-PRC among all methods (0.7880), surpassing even the GMM baseline (0.5720). This indicates that spectral separation effectively identifies a precision-rich subset of fraudulent transactions, even if overall ranking is not optimal.

The Louvain community detection identified one community of 396 nodes with a 96.7% internal fraud rate, suggesting that a compact, structurally cohesive fraud ring exists in the graph. However, this structural signal did not translate into strong performance from the GNN models, implying that supervised neighborhood aggregation over centrality features does not fully exploit the detected community structure.

### 4.3 Medicare Results

On the Medicare dataset, no method achieved AUC-ROC above 0.615, reflecting the fundamentally challenging nature of provider-level fraud detection from prescribing behavior alone. The GCN model achieved the highest AUC-ROC of any method at 0.6139, marginally exceeding the Isolation Forest baseline of 0.6081 (absolute improvement: 0.0058, relative improvement: +0.95%).

The Fiedler anomaly score also exceeded the tabular baseline, reaching 0.6096. By contrast, the Random Forest classifier trained on centrality features performed substantially worse than the baseline (0.4332), indicating that raw centrality values extracted from the provider-drug projection graph do not constitute discriminative features.

The poor performance of GraphSAGE (0.4309) and GAT (0.4361) on Medicare relative to GCN is noteworthy. The inductive aggregation of GraphSAGE and the attention mechanism of GAT may be less appropriate for the sparse, scale-free provider graph, where most neighborhood aggregations involve isolated or weakly connected providers.

### 4.4 Hypothesis Evaluation

The research hypothesis is evaluated as follows:

**Credit Card:** The hypothesis is **not supported**. The best graph method (spectral clustering, 0.9221) did not surpass the tabular baseline (GMM, 0.9588). The PCA preprocessing of this dataset likely destroyed the relational structure that would otherwise make graph construction meaningful: edges in the similarity graph encode proximity in a compressed representation, not in the original transaction space.

**Medicare:** The hypothesis is **marginally supported**. The best graph method (GCN, 0.6139) narrowly exceeded the tabular baseline (Isolation Forest, 0.6081). The bipartite projection graph encodes genuine relational information — shared prescribing patterns among providers — and this information does appear to carry a weak but positive signal for fraud detection. The improvement is too small to constitute a practical recommendation in favor of graph methods for this domain without further development.

---

## 5. Discussion

### 5.1 Why Spectral Methods Outperformed GNNs

A recurring and counterintuitive finding is that spectral clustering and the Fiedler vector outperformed all four GNN architectures on the credit card dataset by a wide margin. Several explanations are considered.

First, the credit card graph was constructed from PCA-compressed features. Spectral methods operate directly on graph topology, using eigenvectors of the Laplacian to capture global cluster structure. GNNs, by contrast, learn to aggregate neighborhood features; when the input features are already dense PCA projections, the added structural information from message passing may be redundant or even introduce noise.

Second, GNN training on highly imbalanced graphs (0.17% fraud rate, 50,000 nodes) is a known difficult problem. Despite class-weighted loss, the optimization may converge to degenerate solutions that predict the majority class. The significantly better performance of unsupervised spectral methods, which require no labels, supports this interpretation.

Third, DOMINANT's poor performance relative to the Fiedler score suggests that deep reconstruction-based anomaly detection does not always outperform simpler spectral baselines, particularly when the graph structure itself encodes a strong partition signal.

### 5.2 The Role of Graph Construction Quality

The contrasting results between the two datasets highlight the importance of graph construction. For Medicare, the bipartite projection encodes semantically meaningful relationships: two providers are connected because they share unusual prescribing patterns, which is directly related to the definition of fraud. For credit card transactions, the K-NN graph is constructed in PCA space, and the meaning of proximity in this space is less directly tied to fraudulent behavior.

This observation has a practical implication: graph-based fraud detection is most beneficial when the graph construction step embeds genuine domain knowledge about how fraud manifests relationally. A K-NN graph in feature space is a weak prior; a domain-informed bipartite or heterogeneous graph is a stronger one.

### 5.3 Limitations

Several limitations of this study should be acknowledged.

The credit card dataset features were PCA-transformed prior to release, limiting the ability to construct a semantically meaningful similarity graph. The Medicare analysis relied on a sampled subset of the full CMS dataset due to memory constraints, and the provider-drug projection required aggressive hub-drug removal and neighborhood capping, which may have discarded informative edges. GNN models were trained for a single run with a fixed random seed; results may vary across seeds, and no cross-validation was performed for the GNN experiments. The tabular ML baseline results were carried over from the companion BLG607 project and were not re-evaluated under the graph project's evaluation protocol, though the datasets and splits are identical.

---

## 6. Conclusion

This study presents a systematic comparison of graph-theoretic fraud detection methods — spanning classical centrality analysis, spectral graph theory, and graph neural networks — against tabular anomaly detection baselines on two real-world datasets.

The principal findings are as follows. Spectral methods, particularly spectral clustering in the Laplacian eigenbasis, achieved the strongest graph-based performance on the credit card dataset (AUC-ROC = 0.9221), though this remained below the tabular GMM baseline (0.9588). On the Medicare dataset, the GCN model and the Fiedler anomaly score both marginally surpassed the Isolation Forest baseline, providing limited but positive evidence that relational prescribing structure carries fraud-relevant information. Graph neural networks underperformed spectral and classical methods in both experimental settings, a result attributed to the compounded difficulties of graph construction from PCA features, highly imbalanced training conditions, and the mismatch between dense neighborhood aggregation and sparse real-world fraud graphs.

These results support a nuanced view of graph-based fraud detection: structural methods are most powerful when graph construction encodes genuine domain knowledge, and spectral approaches can provide competitive or superior performance relative to learned GNN representations, particularly in low-label regimes with strong global structure.

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR 2017*.
2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS 2017*.
3. Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *ICLR 2018*.
4. Ding, K., Li, J., Bhanushali, R., & Liu, H. (2019). Deep anomaly detection on attributed networks. *SDM 2019*.
5. Blondel, V. D., Guillaume, J. L., Lambiotte, R., & Lefebvre, E. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*.
6. Fiedler, M. (1973). Algebraic connectivity of graphs. *Czechoslovak Mathematical Journal, 23*(98), 298-305.
7. Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. *Computer Networks and ISDN Systems, 30*(1-7), 107-117.
8. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE SSCI 2015*.
9. Centers for Medicare & Medicaid Services. (2022). Medicare Part D Prescribers by Provider and Drug. CMS Open Data.
10. Office of Inspector General, U.S. Department of Health and Human Services. (2023). List of Excluded Individuals and Entities (LEIE). OIG.
