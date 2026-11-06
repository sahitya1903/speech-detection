"""
P1 Sequential Baseline: BiLSTM over 3 s segment feature sequences.

Trains through the exact same speaker-independent 5-fold CV harness with
StandardScaler fitted inside each training fold, weighted BCE loss, and
early stopping on validation UAR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

from src.data.build_graph import select_node_features
from src.train.cv import compute_metrics

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "end_review"


class _SeqDataset(Dataset):
    def __init__(self, seqs: list[torch.Tensor], labels: list[int]):
        self.seqs = seqs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.labels[idx]


def _collate_seqs(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    feat_dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, feat_dim, dtype=torch.float32)
    for i, s in enumerate(seqs):
        padded[i, : s.shape[0]] = s
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


class BiLSTMClassifier(nn.Module):
    """2-layer Bidirectional LSTM with masked mean+max pooling over segment sequences."""

    def __init__(self, in_dim: int = 158, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=x.shape[1])

        mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()
        mean_pool = (out * mask_f).sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1.0)
        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        max_pool = out_masked.max(dim=1).values

        feat = torch.cat([mean_pool, max_pool], dim=-1)
        return self.head(feat).squeeze(-1)


def run_bilstm_cv(
    cache_dict: dict[int, dict[str, Any]],
    folds_df: pd.DataFrame,
    feature_set: str = "covarep",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 42,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[dict[str, float]]]]:
    """Train and evaluate BiLSTM through the 5-fold speaker-independent CV harness."""
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    fold_rows = []
    oof_rows = []
    histories: dict[int, list[dict[str, float]]] = {}

    for f in sorted(folds_df["fold"].unique()):
        tr_df = folds_df[folds_df["fold"] != f]
        va_df = folds_df[folds_df["fold"] == f]

        # Fit scaler strictly on training-fold segments
        tr_all_segs = np.concatenate([select_node_features(cache_dict[sid], feature_set).numpy() for sid in tr_df["subject_id"]], axis=0)
        scaler = StandardScaler().fit(tr_all_segs)

        tr_seqs = [torch.from_numpy(scaler.transform(select_node_features(cache_dict[sid], feature_set).numpy()).astype(np.float32)) for sid in tr_df["subject_id"]]
        va_seqs = [torch.from_numpy(scaler.transform(select_node_features(cache_dict[sid], feature_set).numpy()).astype(np.float32)) for sid in va_df["subject_id"]]
        tr_labels = tr_df["label"].tolist()
        va_labels = va_df["label"].tolist()

        tr_loader = DataLoader(_SeqDataset(tr_seqs, tr_labels), batch_size=batch_size, shuffle=True, collate_fn=_collate_seqs)
        va_loader = DataLoader(_SeqDataset(va_seqs, va_labels), batch_size=batch_size, shuffle=False, collate_fn=_collate_seqs)

        n_pos = sum(tr_labels)
        n_neg = len(tr_labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32)

        in_dim = tr_seqs[0].shape[1]
        model = BiLSTMClassifier(in_dim=in_dim)
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
            for x_b, len_b, y_b in tr_loader:
                optimizer.zero_grad()
                logits = model(x_b, len_b)
                loss = criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += float(loss.item()) * len(y_b)
            scheduler.step()
            tr_loss /= len(tr_labels)

            model.eval()
            va_probs_list = []
            with torch.no_grad():
                for x_b, len_b, _ in va_loader:
                    probs = torch.sigmoid(model(x_b, len_b)).numpy()
                    va_probs_list.append(probs)
            va_probs = np.concatenate(va_probs_list)
            m = compute_metrics(np.array(va_labels), va_probs)
            hist.append({"epoch": ep, "train_loss": tr_loss, "val_uar": m["uar"], "val_auc": m["roc_auc"]})

            if m["uar"] > best_uar:
                best_uar = m["uar"]
                best_probs = va_probs.copy()
                best_state = copy.deepcopy(model.state_dict())

        histories[int(f)] = hist
        best_m = compute_metrics(np.array(va_labels), best_probs)
        best_m["model"] = "BiLSTM"
        best_m["fold"] = int(f)
        fold_rows.append(best_m)

        for sid, y_t, p in zip(va_df["subject_id"], va_labels, best_probs):
            oof_rows.append({"model": "BiLSTM", "subject_id": int(sid), "fold": int(f), "y_true": int(y_t), "y_prob": float(p)})

        torch.save(best_state, results_dir / f"bilstm_fold{f}.pt")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    fold_df.to_csv(results_dir / "bilstm_cv.csv", index=False)
    oof_df.to_csv(results_dir / "bilstm_oof.csv", index=False)
    return fold_df, oof_df, histories
