"""
P1 Classical Baselines & Feature-Group Ablation (SVM, RF, LogReg, Gaussian NB).

Evaluated through the speaker-independent 5-fold CV harness with StandardScaler
fitted strictly inside each training fold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.build_graph import select_node_features
from src.train.cv import compute_metrics

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "results" / "end_review"


def _get_classifiers(seed: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {
        "SVM (RBF)": lambda: SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True, random_state=seed),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        "Gaussian NB": lambda: GaussianNB(),
    }
    try:
        from sklearn.ensemble import RandomForestClassifier
        models["Random Forest"] = lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    except ImportError:
        pass
    return models


def run_svm_rf_cv(
    cache_dict: dict[int, dict[str, Any]],
    folds_df: pd.DataFrame,
    feature_set: str = "covarep",
    results_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 5-fold CV for classical baselines on segment-mean cached features."""
    results_dir.mkdir(parents=True, exist_ok=True)

    sids = folds_df["subject_id"].tolist()
    X_all = np.stack([select_node_features(cache_dict[sid], feature_set=feature_set).numpy().mean(axis=0) for sid in sids], axis=0)
    y_all = folds_df["label"].to_numpy(dtype=int)
    folds = folds_df["fold"].to_numpy(dtype=int)

    models = _get_classifiers(seed=seed)
    fold_rows = []
    oof_rows = []

    for name, make_clf in models.items():
        oof_prob = np.zeros(len(sids), dtype=float)
        for f in sorted(np.unique(folds)):
            tr_mask = folds != f
            va_mask = folds == f

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_all[tr_mask])
            X_va = scaler.transform(X_all[va_mask])

            clf = make_clf()
            clf.fit(X_tr, y_all[tr_mask])
            probs = clf.predict_proba(X_va)[:, 1]
            oof_prob[va_mask] = probs

            m = compute_metrics(y_all[va_mask], probs)
            m["model"] = name
            m["feature_set"] = feature_set
            m["fold"] = int(f)
            fold_rows.append(m)

        for sid, y_t, p, f in zip(sids, y_all, oof_prob, folds):
            oof_rows.append({"model": name, "feature_set": feature_set, "subject_id": sid, "fold": int(f), "y_true": int(y_t), "y_prob": float(p)})

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    fold_df.to_csv(results_dir / "svm_rf_cv.csv", index=False)
    oof_df.to_csv(results_dir / "svm_rf_oof.csv", index=False)
    return fold_df, oof_df


def run_feature_ablation_cv(
    cache_dict: dict[int, dict[str, Any]],
    folds_df: pd.DataFrame,
    feature_sets: tuple[str, ...] = ("covarep", "egemaps", "set_a", "set_a_c", "all"),
    results_dir: Path = RESULTS_DIR,
    seed: int = 42,
) -> pd.DataFrame:
    """Run P1 feature-group ablation across classical models (Set A vs Set A+C vs COVAREP)."""
    results_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for fs in feature_sets:
        fdf, _ = run_svm_rf_cv(cache_dict, folds_df, feature_set=fs, results_dir=results_dir, seed=seed)
        grp = fdf.groupby(["feature_set", "model"])[["uar", "macro_f1", "f1_dep", "roc_auc"]].agg(["mean", "std"])
        grp.columns = ["_".join(c) for c in grp.columns]
        all_rows.append(grp.reset_index())

    ablation_df = pd.concat(all_rows, ignore_index=True)
    ablation_df.to_csv(results_dir / "feature_ablation_cv.csv", index=False)
    return ablation_df
