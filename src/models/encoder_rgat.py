"""
P2 Multi-Relational Graph Attention Encoder (Pure-PyTorch RGATConv + Residual + LayerNorm).

Implements multi-head relational graph attention over E_seq (0) and E_knn (1)
without external C-extension dependencies, making it 100% TorchScript-traceable.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Numerically stable per-destination-node softmax over edges."""
    # scores: [E, H]
    max_per_node = torch.full((num_nodes, scores.shape[1]), -1e9, device=scores.device, dtype=scores.dtype)
    max_per_node.index_reduce_(0, dst, scores, reduce="amax", include_self=True)
    exp_scores = torch.exp(scores - max_per_node[dst])
    sum_per_node = torch.zeros((num_nodes, scores.shape[1]), device=scores.device, dtype=scores.dtype)
    sum_per_node.index_add_(0, dst, exp_scores)
    return exp_scores / (sum_per_node[dst] + 1e-12)


class RelationalGATLayer(nn.Module):
    """Multi-head Relational Graph Attention layer (RGATConv equivalent)."""

    def __init__(self, hidden_dim: int, num_relations: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden_dim % heads == 0
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.num_relations = num_relations

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        q = self.q_proj(x).view(n, self.heads, self.head_dim)
        k = self.k_proj(x).view(n, self.heads, self.head_dim)
        v = self.v_proj(x).view(n, self.heads, self.head_dim)
        r = self.rel_emb(edge_type).view(-1, self.heads, self.head_dim)

        # Scaled dot-product attention conditioned on relation embedding
        scores = (q[dst] * (k[src] + r)).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        alpha = self.attn_drop(edge_softmax(scores, dst, n))

        msg = alpha.unsqueeze(-1) * (v[src] + r)  # [E, H, D]
        out = torch.zeros(n, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        return self.out_proj(out.reshape(n, self.hidden_dim))


class RGATEncoder(nn.Module):
    """2-layer RGAT encoder over E_seq (0) and E_knn (1) relations."""

    def __init__(
        self,
        in_dim: int = 158,
        hidden_dim: int = 128,
        num_relations: int = 2,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.convs = nn.ModuleList(
            [RelationalGATLayer(hidden_dim, num_relations=num_relations, heads=heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, edge_index, edge_type)
            h = norm(h + h_res)
            h = self.drop(self.act(h))
        return h
