"""
P0 Data Preprocessing & Cache Builder for DAIC-WOZ (COVAREP + FORMANT CSV path).

Loads shipped 100 Hz COVAREP (74-dim) and FORMANT (5-dim) CSVs, strips all
Ellie turns using transcript timestamps, segments participant speech into
3.0 s windows (1.5 s hop, capped at 256 nodes/subject), computes per-window
mean + std features (voiced-only for F0), and caches one .pt file per subject.

Usage:
    python -m src.data.preprocess
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
DAIC_DIR = ROOT / "DAIC_WOZ"
CACHE_DIR = ROOT / "data" / "cache" / "DAIC_WOZ"

SAMPLE_RATE_HZ = 100
WIN_FRAMES = 300   # 3.0 s at 100 Hz
HOP_FRAMES = 150   # 1.5 s hop (50% overlap)
DEFAULT_MAX_NODES = 256


def load_all_splits(daic_dir: Path = DAIC_DIR) -> pd.DataFrame:
    """Load train, dev, and full_test split CSVs into a unified DataFrame."""
    splits_dir = daic_dir / "splits"
    frames = []

    for split_name, fname, bin_col, score_col in [
        ("train", "train_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
        ("dev", "dev_split_Depression_AVEC2017.csv", "PHQ8_Binary", "PHQ8_Score"),
        ("test", "full_test_split.csv", "PHQ_Binary", "PHQ_Score"),
    ]:
        df = pd.read_csv(splits_dir / fname).dropna(subset=["Participant_ID"])
        out = pd.DataFrame({
            "subject_id": df["Participant_ID"].astype(int),
            "label": df[bin_col].astype(int),
            "phq_score": df[score_col].astype(int),
            "gender": df["Gender"].astype(int),
            "official_split": split_name,
        })
        frames.append(out)

    meta = pd.concat(frames, ignore_index=True).sort_values("subject_id").reset_index(drop=True)
    return meta


def load_participant_turns(transcript_path: Path) -> pd.DataFrame:
    """Return strictly Participant turns (Ellie turns stripped) sorted by start_time."""
    df = pd.read_csv(transcript_path, sep="\t")
    df = df[df["speaker"].astype(str).str.strip().str.lower() == "participant"].copy()
    df = df.dropna(subset=["start_time", "stop_time"]).sort_values("start_time").reset_index(drop=True)
    return df


def extract_subject_windows(
    subject_id: int,
    daic_dir: Path = DAIC_DIR,
    win_frames: int = WIN_FRAMES,
    hop_frames: int = HOP_FRAMES,
    max_nodes: int = DEFAULT_MAX_NODES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract [n_nodes, 158] window features, [n_nodes, 2] timestamps, [n_nodes] turn_ids."""
    cov_path = daic_dir / "covarep" / f"{subject_id}_COVAREP.csv"
    fmt_path = daic_dir / "formant" / f"{subject_id}_FORMANT.csv"
    trn_path = daic_dir / "transcripts" / f"{subject_id}_TRANSCRIPT.csv"

    cov = np.loadtxt(cov_path, delimiter=",", dtype=np.float32)
    fmt = np.loadtxt(fmt_path, delimiter=",", dtype=np.float32)
    n_frames = min(len(cov), len(fmt))
    feats = np.concatenate([cov[:n_frames], fmt[:n_frames]], axis=1)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    turns = load_participant_turns(trn_path)
    part_indices = []
    part_turn_ids = []
    for turn_idx, row in turns.iterrows():
        s_idx = max(0, int(round(float(row["start_time"]) * SAMPLE_RATE_HZ)))
        e_idx = min(n_frames, int(round(float(row["stop_time"]) * SAMPLE_RATE_HZ)))
        if e_idx > s_idx:
            idx_range = np.arange(s_idx, e_idx, dtype=np.int64)
            part_indices.append(idx_range)
            part_turn_ids.append(np.full(len(idx_range), turn_idx, dtype=np.int64))

    if not part_indices:
        raise ValueError(f"Subject {subject_id}: no participant frames found.")

    all_idx = np.concatenate(part_indices)
    all_tids = np.concatenate(part_turn_ids)
    total = len(all_idx)

    # Build sliding windows over participant-only frames
    starts = list(range(0, max(1, total - win_frames + 1), hop_frames))
    if len(starts) > max_nodes:
        sel = np.linspace(0, len(starts) - 1, max_nodes, dtype=int)
        starts = [starts[i] for i in sel]

    node_feats = []
    node_times = []
    node_tids = []

    for st in starts:
        en = min(total, st + win_frames)
        w_idx = all_idx[st:en]
        w_data = feats[w_idx]  # [L, 79]

        mean_vec = w_data.mean(axis=0)
        std_vec = w_data.std(axis=0)

        # Voiced-only F0 (col 0) using VUV mask (col 1 > 0.5)
        voiced = w_data[:, 1] > 0.5
        if voiced.any():
            mean_vec[0] = w_data[voiced, 0].mean()
            std_vec[0] = w_data[voiced, 0].std()
        else:
            mean_vec[0] = 0.0
            std_vec[0] = 0.0

        node_feats.append(np.concatenate([mean_vec, std_vec], axis=0))
        node_times.append([w_idx[0] / SAMPLE_RATE_HZ, w_idx[-1] / SAMPLE_RATE_HZ])
        node_tids.append(int(np.median(all_tids[st:en])))

    x = torch.from_numpy(np.stack(node_feats, axis=0).astype(np.float32))
    timestamps = torch.tensor(node_times, dtype=torch.float32)
    turn_ids = torch.tensor(node_tids, dtype=torch.long)
    return x, timestamps, turn_ids


def build_cache(
    daic_dir: Path = DAIC_DIR,
    cache_dir: Path = CACHE_DIR,
    max_nodes: int = DEFAULT_MAX_NODES,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Process all DAIC-WOZ subjects and write <subject_id>.pt + metadata.csv."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = load_all_splits(daic_dir)
    records = []

    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Caching DAIC-WOZ"):
        out_path = cache_dir / f"{row.subject_id}.pt"
        if out_path.exists() and not overwrite:
            payload = torch.load(out_path, weights_only=False)
            n_nodes = int(payload["x"].shape[0])
        else:
            x, timestamps, turn_ids = extract_subject_windows(
                row.subject_id, daic_dir=daic_dir, max_nodes=max_nodes
            )
            payload = {
                "x": x,
                "timestamps": timestamps,
                "turn_ids": turn_ids,
                "y": torch.tensor(row.label, dtype=torch.long),
                "phq_score": int(row.phq_score),
                "gender": int(row.gender),
                "subject_id": int(row.subject_id),
                "official_split": str(row.official_split),
            }
            torch.save(payload, out_path)
            n_nodes = int(x.shape[0])

        records.append({
            "subject_id": row.subject_id,
            "label": row.label,
            "phq_score": row.phq_score,
            "gender": row.gender,
            "official_split": row.official_split,
            "n_nodes": n_nodes,
        })

    summary = pd.DataFrame(records)
    summary.to_csv(cache_dir / "metadata.csv", index=False)
    print(f"Cached {len(summary)} subjects to {cache_dir} (avg nodes: {summary['n_nodes'].mean():.1f})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DAIC-WOZ .pt cache from COVAREP+FORMANT CSVs.")
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cache(max_nodes=args.max_nodes, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
