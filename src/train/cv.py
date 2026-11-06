"""
P1 Speaker-Independent 5-Fold CV Harness & Evaluation Metrics.

Stratifies on (label x gender) across all 189 DAIC-WOZ subjects and saves
data/cache/DAIC_WOZ/folds_5cv.csv so every baseline and GNN uses the exact
same speaker-independent folds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "data" / "cache" / "DAIC_WOZ"
FOLDS_CSV = CACHE_DIR / "folds_5cv.csv"


def build_or_load_folds(
    cache_dir: Path = CACHE_DIR,
    out_csv: Path = FOLDS_CSV,
    n_splits: int = 5,
    seed: int = 42,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Create or load 5-fold speaker-independent splits stratified by label x gender."""
    if out_csv.exists() and not overwrite:
        df = pd.read_csv(out_csv)
    else:
        meta = pd.read_csv(cache_dir / "metadata.csv").sort_values("subject_id").reset_index(drop=True)
        strat_key = meta["label"].astype(int) * 2 + meta["gender"].astype(int)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        meta["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(skf.split(meta, strat_key)):
            meta.loc[val_idx, "fold"] = fold_idx
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(out_csv, index=False)
        df = meta

    assert_no_speaker_leakage(df)
    return df


def assert_no_speaker_leakage(folds_df: pd.DataFrame) -> None:
    """Verify every subject appears in exactly one fold."""
    assert folds_df["subject_id"].is_unique, "Duplicate subject_ids found in folds!"
    assert (folds_df["fold"] >= 0).all(), "Unassigned subjects found in folds!"
    for f in sorted(folds_df["fold"].unique()):
        train_ids = set(folds_df.loc[folds_df["fold"] != f, "subject_id"])
        val_ids = set(folds_df.loc[folds_df["fold"] == f, "subject_id"])
        overlap = train_ids & val_ids
        assert not overlap, f"Speaker leakage in fold {f}: {overlap}"


def load_cached_subjects(cache_dir: Path = CACHE_DIR) -> dict[int, dict[str, Any]]:
    """Load all cached <subject_id>.pt files into memory."""
    cache = {}
    for p in sorted(cache_dir.glob("*.pt")):
        payload = torch.load(p, weights_only=False)
        cache[int(payload["subject_id"])] = payload
    return cache


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute UAR, macro-F1, depressed F1/recall/precision, ROC-AUC, and accuracy."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_dep": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_dep": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "prec_dep": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def bootstrap_uar_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Return (point_uar, ci_low, ci_high) via stratified-like percentile bootstrap."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    n = len(y_true)
    point = float(balanced_accuracy_score(y_true, y_pred))

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))

    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    return point, float(ci_low), float(ci_high)
