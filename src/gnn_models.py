"""
gnn_models.py
=============
Graph Neural Network models using PyTorch Geometric:
  - GCN (Graph Convolutional Network)
  - GraphSAGE
  - GAT (Graph Attention Network)
  - DOMINANT (Deep Anomaly Detection on Attributed Networks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# GCN
# ============================================================

class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network (Kipf & Welling, 2017).
    For node classification (binary: fraud vs normal).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        return torch.sigmoid(out).squeeze(-1)

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================
# GraphSAGE
# ============================================================

class GraphSAGE(nn.Module):
    """
    GraphSAGE: Inductive Representation Learning (Hamilton et al., 2017).
    Uses mean aggregation of neighborhood embeddings.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 aggr: str = "mean", dropout: float = 0.5):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2, aggr=aggr)
        self.classifier = nn.Linear(hidden_channels // 2, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.classifier(x)
        return torch.sigmoid(out).squeeze(-1)

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================
# GAT
# ============================================================

class GAT(nn.Module):
    """
    Graph Attention Network (Veličković et al., 2018).
    Uses multi-head attention to weight neighbor contributions.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 8,
                 heads: int = 4, dropout: float = 0.6):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                              dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1,
                              dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        out = self.classifier(x)
        return torch.sigmoid(out).squeeze(-1)

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================
# DOMINANT — Deep Anomaly Detection on Attributed Networks
# ============================================================

class DOMINANTEncoder(nn.Module):
    """Shared GCN encoder for DOMINANT."""

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 embedding_dim: int = 32, dropout: float = 0.3):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class DOMINANTAttributeDecoder(nn.Module):
    """Attribute reconstruction decoder for DOMINANT."""

    def __init__(self, embedding_dim: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.3):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(embedding_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, z, edge_index):
        z = self.conv1(z, edge_index)
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.conv2(z, edge_index)
        return z


class DOMINANT(nn.Module):
    """
    DOMINANT: Deep Anomaly Detection on Attributed Networks (Ding et al., 2019).
    
    Unsupervised anomaly detection via:
    - Node attribute reconstruction error
    - Graph structure reconstruction error (inner product decoder)
    
    Anomaly score = alpha * struct_error + (1-alpha) * attr_error
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 embedding_dim: int = 32, dropout: float = 0.3, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.encoder = DOMINANTEncoder(in_channels, hidden_channels, embedding_dim, dropout)
        self.attr_decoder = DOMINANTAttributeDecoder(embedding_dim, hidden_channels,
                                                      in_channels, dropout)

    def forward(self, x, edge_index):
        # Encode
        z = self.encoder(x, edge_index)

        # Attribute reconstruction
        x_hat = self.attr_decoder(z, edge_index)

        # Structure reconstruction (inner product)
        adj_hat = torch.sigmoid(torch.mm(z, z.t()))

        return x_hat, adj_hat, z

    def compute_anomaly_scores(self, x, x_hat, adj_hat, adj_dense):
        """
        Compute node-level anomaly scores.

        attr_error:   reconstruction error for node features
        struct_error: reconstruction error for adjacency structure
        """
        attr_error = torch.mean((x - x_hat) ** 2, dim=1)
        struct_error = torch.mean((adj_dense - adj_hat) ** 2, dim=1)

        scores = self.alpha * struct_error + (1 - self.alpha) * attr_error
        return scores


# ============================================================
# Training & Evaluation Utilities
# ============================================================

class GNNTrainer:
    """
    Generic trainer for supervised GNN models (GCN, GraphSAGE, GAT).
    """

    def __init__(self, model, config: dict, device: str = None):
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def train_epoch(self, data, optimizer, criterion):
        self.model.train()
        optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, mask_name: str = "test_mask") -> dict:
        self.model.eval()
        mask = getattr(data, mask_name)
        out = self.model(data.x, data.edge_index)
        scores = out[mask].cpu().numpy()
        labels = data.y[mask].cpu().numpy()

        auc = roc_auc_score(labels, scores) if labels.sum() > 0 else 0.0
        ap = average_precision_score(labels, scores) if labels.sum() > 0 else 0.0

        # Optimal F1 threshold
        thresholds = np.linspace(0, 1, 100)
        best_f1, best_thresh = 0, 0.5
        for t in thresholds:
            preds = (scores >= t).astype(int)
            if preds.sum() > 0:
                f1 = f1_score(labels, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = t

        return {"auc_roc": auc, "auc_prc": ap, "f1": best_f1, "threshold": best_thresh}

    def fit(self, data, epochs: int = 200, lr: float = 0.01,
            weight_decay: float = 5e-4, verbose: bool = True) -> list:
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)

        # Class-weighted loss
        n_fraud = data.y.sum().item()
        n_normal = len(data.y) - n_fraud
        pos_weight = torch.tensor([n_normal / (n_fraud + 1e-8)]).to(self.device)
        criterion = nn.BCELoss(reduction="mean")  # using sigmoid already applied

        history = []
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(data, optimizer, criterion)
            if epoch % 20 == 0 or epoch == epochs:
                val_metrics = self.evaluate(data, "val_mask")
                history.append({"epoch": epoch, "train_loss": loss, **val_metrics})
                if verbose:
                    print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | "
                          f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                          f"Val AP: {val_metrics['auc_prc']:.4f}")

        return history


def prepate_pyg_data(G, node_features_df=None, seed: int = 42,
                     train_ratio: float = 0.7, val_ratio: float = 0.1):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Features: either from node attributes or from the DataFrame.
    """
    import torch
    from torch_geometric.data import Data

    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    # Labels
    y = torch.tensor([G.nodes[n].get("label", 0) for n in nodes], dtype=torch.long)

    # Features
    if node_features_df is not None:
        # Align with node order
        feat_cols = [c for c in node_features_df.columns
                     if c not in ["node_id", "label"]]
        node_features_df = node_features_df.set_index("node_id")
        X = np.zeros((len(nodes), len(feat_cols)))
        for i, n in enumerate(nodes):
            if n in node_features_df.index:
                X[i] = node_features_df.loc[n, feat_cols].values
    else:
        # Use node features attribute if available, else degree as single feature
        node_feats = [G.nodes[n].get("features", [G.degree(n)]) for n in nodes]
        max_len = max(len(f) for f in node_feats)
        X = np.zeros((len(nodes), max_len))
        for i, f in enumerate(node_feats):
            X[i, :len(f)] = f

    x = torch.tensor(X, dtype=torch.float)

    # Edges
    edge_list = [(node_index[u], node_index[v]) for u, v in G.edges()]
    edge_list += [(v, u) for u, v in edge_list]  # undirected → both directions
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Masks (stratified split)
    np.random.seed(seed)
    n = len(nodes)
    indices = np.arange(n)

    fraud_idx = np.where(y.numpy() == 1)[0]
    normal_idx = np.where(y.numpy() == 0)[0]

    def split_idx(idxs):
        np.random.shuffle(idxs)
        n_train = int(len(idxs) * train_ratio)
        n_val = int(len(idxs) * val_ratio)
        return idxs[:n_train], idxs[n_train:n_train + n_val], idxs[n_train + n_val:]

    f_train, f_val, f_test = split_idx(fraud_idx.copy())
    n_train, n_val, n_test = split_idx(normal_idx.copy())

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[np.concatenate([f_train, n_train])] = True
    val_mask[np.concatenate([f_val, n_val])] = True
    test_mask[np.concatenate([f_test, n_test])] = True

    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    print(f"[PyG Data] Nodes: {n}, Edges: {edge_index.shape[1] // 2}, "
          f"Features: {X.shape[1]}, Fraud: {y.sum().item()}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    return data
