"""
P2 Flat-Readout RGAT Classifier + CV Training + TorchScript Export Smoke-Test.

Readout concatenates mean-pool, max-pool, and learned attention-pool -> [384] -> MLP -> logit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from src.data.build_graph import build_subject_graph, collate_graphs, select_node_features
from src.models.encoder_rgat import RGATEncoder, edge_softmax
from src.train.cv import compute_metrics

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "end_review"


class DepressionRGATClassifier(nn.Module):
    """P2 Baseline GNN: RGAT encoder + flat readout (mean/max/attention) + MLP."""

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
        self.hidden_dim = hidden_dim
        self.encoder = RGATEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.att_gate = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int = 1,
    ) -> torch.Tensor:
        h = self.encoder(x, edge_index, edge_type)

        # Mean pool
        counts = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones(h.shape[0], 1, device=h.device, dtype=h.dtype))
        p_mean = torch.zeros(num_graphs, self.hidden_dim, device=h.device, dtype=h.dtype)
        p_mean.index_add_(0, batch, h)
        p_mean = p_mean / counts.clamp(min=1.0)

        # Max pool
        p_max = torch.full((num_graphs, self.hidden_dim), -1e9, device=h.device, dtype=h.dtype)
        p_max.index_reduce_(0, batch, h, reduce="amax", include_self=True)

        # Attention pool
        att_weights = edge_softmax(self.att_gate(h), batch, num_graphs)
        p_att = torch.zeros(num_graphs, self.hidden_dim, device=h.device, dtype=h.dtype)
        p_att.index_add_(0, batch, att_weights * h)

        graph_emb = torch.cat([p_mean, p_max, p_att], dim=-1)
        return self.head(graph_emb).squeeze(-1)


def run_gnn_cv(
    cache_dict: dict[int, dict[str, Any]],
    folds_df: pd.DataFrame,
    feature_set: str = "covarep",
    use_recurrence: bool = True,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 7e-4,
    k: int = 5,
    seed: int = 42,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[dict[str, float]]]]:
    """Train and evaluate RGAT GNN through the 5-fold speaker-independent CV harness."""
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_relations = 3 if use_recurrence else 2
    fold_rows = []
    oof_rows = []
    histories: dict[int, list[dict[str, float]]] = {}

    for f in sorted(folds_df["fold"].unique()):
        tr_df = folds_df[folds_df["fold"] != f]
        va_df = folds_df[folds_df["fold"] == f]

        tr_all_segs = np.concatenate([select_node_features(cache_dict[sid], feature_set).numpy() for sid in tr_df["subject_id"]], axis=0)
        scaler = StandardScaler().fit(tr_all_segs)

        tr_graphs = [build_subject_graph(cache_dict[sid], k=k, scaler=scaler, feature_set=feature_set, use_recurrence=use_recurrence) for sid in tr_df["subject_id"]]
        va_graphs = [build_subject_graph(cache_dict[sid], k=k, scaler=scaler, feature_set=feature_set, use_recurrence=use_recurrence) for sid in va_df["subject_id"]]

        tr_loader = DataLoader(tr_graphs, batch_size=batch_size, shuffle=True, collate_fn=collate_graphs)
        va_loader = DataLoader(va_graphs, batch_size=batch_size, shuffle=False, collate_fn=collate_graphs)

        tr_labels = tr_df["label"].tolist()
        va_labels = va_df["label"].to_numpy(dtype=int)
        n_pos = sum(tr_labels)
        n_neg = len(tr_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)

        in_dim = tr_graphs[0].x.shape[1]
        model = DepressionRGATClassifier(in_dim=in_dim, num_relations=num_relations)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_uar = -1.0
        best_probs = None
        best_state = None
        hist = []

        for ep in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            for batch in tr_loader:
                optimizer.zero_grad()
                logits = model(batch.x, batch.edge_index, batch.edge_type, batch.batch, batch.num_graphs)
                loss = criterion(logits, batch.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += float(loss.item()) * batch.num_graphs
            scheduler.step()
            tr_loss /= len(tr_graphs)

            model.eval()
            va_probs_list = []
            with torch.no_grad():
                for batch in va_loader:
                    probs = torch.sigmoid(
                        model(batch.x, batch.edge_index, batch.edge_type, batch.batch, batch.num_graphs)
                    ).numpy()
                    va_probs_list.append(probs)
            va_probs = np.concatenate(va_probs_list)
            m = compute_metrics(va_labels, va_probs)
            hist.append({"epoch": ep, "train_loss": tr_loss, "val_uar": m["uar"], "val_auc": m["roc_auc"]})

            if m["uar"] > best_uar:
                best_uar = m["uar"]
                best_probs = va_probs.copy()
                best_state = copy.deepcopy(model.state_dict())

        histories[int(f)] = hist
        best_m = compute_metrics(va_labels, best_probs)
        best_m["model"] = "RGAT GNN"
        best_m["fold"] = int(f)
        fold_rows.append(best_m)

        for sid, y_t, p in zip(va_df["subject_id"], va_labels, best_probs):
            oof_rows.append({"model": "RGAT GNN", "subject_id": int(sid), "fold": int(f), "y_true": int(y_t), "y_prob": float(p)})

        torch.save(best_state, results_dir / f"rgat_fold{f}.pt")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    fold_df.to_csv(results_dir / "rgat_cv.csv", index=False)
    oof_df.to_csv(results_dir / "rgat_oof.csv", index=False)
    return fold_df, oof_df, histories


def run_export_smoke_test(
    in_dim: int = 158,
    out_flag_path: Path = ROOT / "results" / "p2_export_smoke_test.ok",
) -> bool:
    """Verify TorchScript trace export of the RGAT classifier with fixed-shape dummy input."""
    model = DepressionRGATClassifier(in_dim=in_dim).eval()
    n_nodes = 256
    x = torch.randn(n_nodes, in_dim)
    edge_index = torch.randint(0, n_nodes, (2, 512), dtype=torch.long)
    edge_type = torch.randint(0, 2, (512,), dtype=torch.long)
    batch = torch.zeros(n_nodes, dtype=torch.long)

    with torch.no_grad():
        ref = model(x, edge_index, edge_type, batch)
        traced = torch.jit.trace(model, (x, edge_index, edge_type, batch))
        out = traced(x, edge_index, edge_type, batch)
        assert torch.allclose(ref, out, atol=1e-5), "TorchScript trace output mismatch!"

    out_flag_path.parent.mkdir(parents=True, exist_ok=True)
    out_flag_path.write_text("TorchScript trace smoke-test passed.\n", encoding="utf-8")
    return True
