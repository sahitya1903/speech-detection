"""
P2 Multi-Relational Graph Construction for DAIC-WOZ Subject Graphs.

Constructs per-subject graphs over 3 s speech segments with three edge types:
  - E_seq (edge_type = 0): temporal edges i <-> i+1 (prosodic rhythm & dynamics)
  - E_knn (edge_type = 1): cosine k-NN similarity edges (k=5) in feature space
  - E_rec (edge_type = 2): recurrence edges (cosine similarity >= threshold among non-adjacent nodes)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


@dataclass
class GraphData:
    """Lightweight graph container matching PyG Data attributes."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    y: torch.Tensor
    subject_id: int
    gender: int

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass
class GraphBatch:
    """Batched graphs for pure-PyTorch GNN training."""
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    batch: torch.Tensor
    y: torch.Tensor
    num_graphs: int


def collate_graphs(graphs: list[GraphData]) -> GraphBatch:
    """Collate a list of GraphData objects into a disjoint union GraphBatch."""
    xs, eis, ets, batches, ys = [], [], [], [], []
    node_offset = 0
    for g_idx, g in enumerate(graphs):
        n = g.num_nodes
        xs.append(g.x)
        eis.append(g.edge_index + node_offset)
        ets.append(g.edge_type)
        batches.append(torch.full((n,), g_idx, dtype=torch.long))
        ys.append(g.y)
        node_offset += n

    return GraphBatch(
        x=torch.cat(xs, dim=0),
        edge_index=torch.cat(eis, dim=1),
        edge_type=torch.cat(ets, dim=0),
        batch=torch.cat(batches, dim=0),
        y=torch.cat(ys, dim=0),
        num_graphs=len(graphs),
    )


def select_node_features(cache_dict: dict[str, Any], feature_set: str = "set_a_c") -> torch.Tensor:
    """Assemble node feature matrix from cached streams.

    feature_set options:
      - 'covarep': 158-dim COVAREP+FORMANT
      - 'egemaps': 88-dim eGeMAPS v02
      - 'set_a':   99-dim eGeMAPS + prosody/CPP/timing
      - 'set_a_c': 107-dim eGeMAPS + prosody + Set C linguistic (default when available)
      - 'all':     265-dim COVAREP+FORMANT + eGeMAPS + prosody + Set C
    """
    if feature_set == "covarep" or "x_egemaps" not in cache_dict:
        return cache_dict["x"]
    if feature_set == "egemaps":
        return cache_dict["x_egemaps"]
    if feature_set == "set_a":
        return torch.cat([cache_dict["x_egemaps"], cache_dict["x_prosody"]], dim=1)
    if feature_set == "set_a_c":
        return torch.cat([cache_dict["x_egemaps"], cache_dict["x_prosody"], cache_dict["x_set_c"]], dim=1)
    if feature_set == "all":
        return torch.cat([cache_dict["x"], cache_dict["x_egemaps"], cache_dict["x_prosody"], cache_dict["x_set_c"]], dim=1)
    raise ValueError(f"Unknown feature_set: {feature_set}")


def build_temporal_edges(n_nodes: int) -> torch.Tensor:
    """Bidirectional sequential temporal edges E_seq: i <-> i+1."""
    if n_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    src = torch.arange(0, n_nodes - 1, dtype=torch.long)
    dst = src + 1
    return torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)


def build_knn_edges(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cosine similarity k-NN edges E_knn (undirected top-k per node, excluding self)."""
    n_nodes = x.shape[0]
    if n_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = min(k, n_nodes - 1)
    x_norm = F.normalize(x, p=2, dim=1)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1e9)
    _, topk_idx = torch.topk(sim, k=k_eff, dim=1)
    src = torch.arange(n_nodes, dtype=torch.long).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    dst = topk_idx.reshape(-1)
    edges = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
    return torch.unique(edges, dim=1)


def build_recurrence_edges(x: torch.Tensor, quantile: float = 0.95, min_lag: int = 3) -> torch.Tensor:
    """Recurrence edges E_rec between non-adjacent segments above a cosine-similarity quantile."""
    n_nodes = x.shape[0]
    if n_nodes <= min_lag + 1:
        return torch.empty((2, 0), dtype=torch.long)
    x_norm = F.normalize(x, p=2, dim=1)
    sim = x_norm @ x_norm.t()
    idx = torch.arange(n_nodes)
    lag_mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() >= min_lag
    valid_sims = sim[lag_mask]
    if valid_sims.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    thresh = torch.quantile(valid_sims, quantile)
    rec_mask = lag_mask & (sim >= thresh)
    src, dst = torch.nonzero(rec_mask, as_tuple=True)
    if src.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def build_subject_graph(
    cache_dict: dict[str, Any],
    k: int = 5,
    scaler: StandardScaler | None = None,
    feature_set: str = "covarep",
    use_recurrence: bool = True,
) -> GraphData:
    """Convert cached subject payload into a multi-relational GraphData object."""
    x_raw = select_node_features(cache_dict, feature_set=feature_set)
    x_np = x_raw.numpy()
    if scaler is not None:
        x_np = scaler.transform(x_np).astype(np.float32)
    x = torch.from_numpy(x_np)

    n_nodes = x.shape[0]
    e_seq = build_temporal_edges(n_nodes)
    e_knn = build_knn_edges(x, k=k)

    edge_parts = [e_seq, e_knn]
    type_parts = [
        torch.zeros(e_seq.shape[1], dtype=torch.long),
        torch.ones(e_knn.shape[1], dtype=torch.long),
    ]
    if use_recurrence:
        e_rec = build_recurrence_edges(x)
        edge_parts.append(e_rec)
        type_parts.append(torch.full((e_rec.shape[1],), 2, dtype=torch.long))

    edge_index = torch.cat(edge_parts, dim=1)
    edge_type = torch.cat(type_parts, dim=0)

    y = cache_dict["y"].view(1).float()
    return GraphData(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=y,
        subject_id=int(cache_dict["subject_id"]),
        gender=int(cache_dict["gender"]),
    )
