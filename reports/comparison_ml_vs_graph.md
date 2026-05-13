# ML vs Graph Methods -- Final Comparison Report

**Course:** Graduate Graph Theory
**Author:** Baha Kirbasoglu

## Comparison Table

| Dataset     | Category        | Method                 |   AUC_ROC |   AUC_PRC |          F1 |   Train_Time_s |
|:------------|:----------------|:-----------------------|----------:|----------:|------------:|---------------:|
| Credit Card | Tabular-ML      | OCSVM                  |  0.9514   | 0.2731    | nan         |      nan       |
| Credit Card | Tabular-ML      | GMM                    |  0.9588   | 0.572     | nan         |      nan       |
| Credit Card | Tabular-ML      | IsolationForest        |  0.9496   | 0.1543    | nan         |      nan       |
| Credit Card | Tabular-ML      | LOF                    |  0.9532   | 0.6391    | nan         |      nan       |
| Credit Card | Tabular-ML      | Ensemble_Best          |  0.9535   | 0.5191    | nan         |      nan       |
| Medicare    | Tabular-ML      | OCSVM                  |  0.5261   | 0.0019    | nan         |      nan       |
| Medicare    | Tabular-ML      | GMM                    |  0.5977   | 0.0023    | nan         |      nan       |
| Medicare    | Tabular-ML      | IsolationForest        |  0.6081   | 0.0038    | nan         |      nan       |
| Medicare    | Tabular-ML      | LOF                    |  0.6043   | 0.0021    | nan         |      nan       |
| Medicare    | Tabular-ML      | Ensemble_Best          |  0.5974   | 0.0037    | nan         |      nan       |
| Credit Card | Graph-Classical | RF_Centrality_Features |  0.883933 | 0.443101  | nan         |      nan       |
| Medicare    | Graph-Classical | RF_Centrality_Features |  0.43322  | 0.0144332 | nan         |      nan       |
| Credit Card | Graph-Spectral  | Fiedler_Anomaly        |  0.886142 | 0.788034  | nan         |      nan       |
| Medicare    | Graph-Spectral  | Fiedler_Anomaly        |  0.609581 | 0.024473  | nan         |      nan       |
| Credit Card | Graph-Spectral  | SpectralClustering_k10 |  0.922057 | 0.748936  | nan         |      nan       |
| Medicare    | Graph-Spectral  | SpectralClustering_k10 |  0.577929 | 0.0251088 | nan         |      nan       |
| Credit Card | Graph-Spectral  | SpectralClustering_k10 |  0.922057 | 0.748936  | nan         |      nan       |
| Medicare    | Graph-Spectral  | SpectralClustering_k10 |  0.577929 | 0.0251088 | nan         |      nan       |
| Credit Card | Graph-Spectral  | SpectralClustering_k10 |  0.922057 | 0.748936  | nan         |      nan       |
| Medicare    | Graph-Spectral  | SpectralClustering_k10 |  0.577929 | 0.0251088 | nan         |      nan       |
| Credit Card | Graph-GNN       | GCN                    |  0.614383 | 0.0693032 |   0.12037   |       35.228   |
| Medicare    | Graph-GNN       | GCN                    |  0.613902 | 0.0273186 |   0.0570571 |        9.67455 |
| Credit Card | Graph-GNN       | GraphSAGE              |  0.517141 | 0.0642901 |   0.096     |       28.3635  |
| Medicare    | Graph-GNN       | GraphSAGE              |  0.430864 | 0.0170853 |   0.0368957 |        7.67709 |
| Credit Card | Graph-GNN       | GAT                    |  0.358587 | 0.0136072 |   0.019602  |       51.8386  |
| Medicare    | Graph-GNN       | GAT                    |  0.436068 | 0.0159017 |   0.0368957 |       16.5301  |
| Credit Card | Graph-GNN-Unsup | DOMINANT               |  0.570824 | 0.0197128 | nan         |     1263.68    |
| Medicare    | Graph-GNN-Unsup | DOMINANT               |  0.550927 | 0.0200126 | nan         |       37.4321  |

## Key Findings
- Medicare: Graph methods marginally outperform tabular baseline (GCN: 0.6139 vs IsoForest: 0.6081, +0.95%); relational prescribing structure provides a weak but positive signal
- Credit Card: Tabular ML outperforms graph methods (GMM: 0.9588 vs Spectral Clustering: 0.9221, -3.83%); PCA preprocessing limited the quality of graph construction
- Note: SpectralClustering_k10 rows contain duplicates in the raw results and should be deduplicated before further analysis
