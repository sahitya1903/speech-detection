"""
P0 Set A Extras & eGeMAPS Extractor (Raw Audio + Transcripts -> Cache).

For each cached subject (<sid>.pt), extracts per-3s-window:
  - eGeMAPSv02 Functionals (88-dim) via openSMILE
  - Prosody, rhythm, pause, and voice-quality extras (11-dim):
      syllable_rate, articulation_rate, speech_to_pause_ratio,
      pause_count, mean_pause_len, pct_silence,
      f0_slope, cpp_mean, cpp_std, words_per_sec, response_latency
Appends `x_egemaps` [n_nodes, 88], `x_prosody` [n_nodes, 11], and `x_set_c` [n_nodes, 8]
to each data/cache/DAIC_WOZ/<sid>.pt file.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import opensmile
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

from src.data.preprocess import CACHE_DIR, DAIC_DIR, load_participant_turns
from src.data.features_linguistic import extract_set_c_for_subject

_SMILE = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def _compute_turn_timing(transcript_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return per-participant-turn (words_per_sec, response_latency) arrays."""
    df = pd.read_csv(transcript_path, sep="\t").dropna(subset=["start_time", "stop_time"])
    df = df.sort_values("start_time").reset_index(drop=True)

    wps_list = []
    lat_list = []
    last_ellie_stop = None

    for row in df.itertuples(index=False):
        spk = str(row.speaker).strip().lower()
        st, en = float(row.start_time), float(row.stop_time)
        if spk == "ellie":
            last_ellie_stop = en
        elif spk == "participant":
            dur = max(0.1, en - st)
            n_words = len(str(row.value).split())
            wps_list.append(n_words / dur)
            lat = max(0.0, st - last_ellie_stop) if last_ellie_stop is not None else 0.0
            lat_list.append(min(lat, 15.0))

    return np.array(wps_list, dtype=np.float32), np.array(lat_list, dtype=np.float32)


def _prosody_segment_features(seg: np.ndarray, sr: int) -> np.ndarray:
    """Compute 9 acoustic prosody/rhythm/CPP features for a 3 s audio window."""
    if len(seg) < int(0.2 * sr) or np.allclose(seg, 0.0):
        return np.zeros(9, dtype=np.float32)

    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=float(sr))
    dur = max(0.1, snd.duration)

    # Intensity & silence/pause stats (threshold: 25 dB below max)
    try:
        intensity = snd.to_intensity(minimum_pitch=75.0)
        vals = intensity.values[0]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 2:
            thresh = np.max(vals) - 25.0
            silent = vals < thresh
            pct_silence = float(np.mean(silent))
            # Count transitions into silence
            transitions = np.diff(silent.astype(int))
            pause_count = float(np.sum(transitions == 1))
            dt = intensity.dx
            mean_pause_len = float(np.sum(silent) * dt / max(1.0, pause_count)) if pause_count > 0 else 0.0
            speaking_time = max(0.05, dur * (1.0 - pct_silence))
            speech_to_pause = float(speaking_time / max(0.05, dur - speaking_time))

            # Syllable nuclei proxy: local peaks in smoothed intensity above threshold
            peaks = (vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:]) & (vals[1:-1] > thresh + 3.0)
            n_syl = float(np.sum(peaks))
            syl_rate = n_syl / dur
            art_rate = n_syl / speaking_time
        else:
            syl_rate = art_rate = speech_to_pause = pause_count = mean_pause_len = pct_silence = 0.0
    except Exception:
        syl_rate = art_rate = speech_to_pause = pause_count = mean_pause_len = pct_silence = 0.0

    # F0 slope
    try:
        pitch = snd.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)
        f0 = pitch.selected_array["frequency"]
        voiced = f0 > 0
        if np.sum(voiced) >= 3:
            t = np.arange(len(f0))[voiced] * pitch.dx
            f0_slope = float(np.polyfit(t, f0[voiced], 1)[0])
        else:
            f0_slope = 0.0
    except Exception:
        f0_slope = 0.0

    # Smoothed Cepstral Peak Prominence (CPPS)
    try:
        cpps = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        cpp_mean = float(call(cpps, "Get CPPS", "no", 0.02, 0.0005, 60.0, 330.0, 0.05, "Parabolic", 0.001, 0.0, "Straight", "Robust"))
        cpp_std = 0.0
        if not np.isfinite(cpp_mean):
            cpp_mean = 0.0
    except Exception:
        cpp_mean = 0.0
        cpp_std = 0.0

    return np.array(
        [syl_rate, art_rate, speech_to_pause, pause_count, mean_pause_len, pct_silence, f0_slope, cpp_mean, cpp_std],
        dtype=np.float32,
    )


def process_subject(
    subject_id: int,
    daic_dir: Path = DAIC_DIR,
    cache_dir: Path = CACHE_DIR,
    overwrite: bool = False,
) -> dict[str, int]:
    """Extract eGeMAPS (88) + prosody (11) + Set C (8) and append to <sid>.pt."""
    pt_path = cache_dir / f"{subject_id}.pt"
    payload = torch.load(pt_path, weights_only=False)
    if not overwrite and "x_egemaps" in payload and "x_prosody" in payload and "x_set_c" in payload:
        return {"subject_id": subject_id, "n_nodes": int(payload["x"].shape[0])}

    wav_path = daic_dir / "audio" / f"{subject_id}_AUDIO.wav"
    trn_path = daic_dir / "transcripts" / f"{subject_id}_TRANSCRIPT.csv"

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    timestamps = payload["timestamps"].numpy()  # [n_nodes, 2]
    turn_ids = payload["turn_ids"]

    wps_arr, lat_arr = _compute_turn_timing(trn_path)
    t_idx = np.clip(turn_ids.numpy(), 0, max(0, len(wps_arr) - 1))

    ege_rows = []
    pro_rows = []

    for i, (t_start, t_end) in enumerate(timestamps):
        s_i = max(0, int(round(t_start * sr)))
        e_i = min(len(audio), int(round(t_end * sr)))
        seg = audio[s_i:e_i]
        if len(seg) < int(0.1 * sr):
            seg = np.zeros(int(0.5 * sr), dtype=np.float32)

        ege = _SMILE.process_signal(seg, sr).to_numpy(dtype=np.float32)[0]
        ege = np.nan_to_num(ege, nan=0.0, posinf=0.0, neginf=0.0)
        ege_rows.append(ege)

        ac_pro = _prosody_segment_features(seg, sr)
        wps_val = float(wps_arr[t_idx[i]]) if len(wps_arr) > 0 else 0.0
        lat_val = float(lat_arr[t_idx[i]]) if len(lat_arr) > 0 else 0.0
        pro_rows.append(np.concatenate([ac_pro, np.array([wps_val, lat_val], dtype=np.float32)]))

    payload["x_egemaps"] = torch.from_numpy(np.stack(ege_rows, axis=0))
    payload["x_prosody"] = torch.from_numpy(np.stack(pro_rows, axis=0))
    payload["x_set_c"] = extract_set_c_for_subject(subject_id, turn_ids, daic_dir=daic_dir)
    torch.save(payload, pt_path)
    return {"subject_id": subject_id, "n_nodes": int(payload["x"].shape[0])}


def _worker(args: tuple[int, bool]) -> dict[str, int]:
    sid, overwrite = args
    return process_subject(sid, overwrite=overwrite)


def main() -> None:
    parser = argparse.ArgumentParser(description="Append eGeMAPS + Set A prosody + Set C linguistic features to cache.")
    parser.add_argument("--subject", type=int, default=None, help="Process a single subject ID (for benchmarking).")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.subject is not None:
        res = process_subject(args.subject, overwrite=True)
        print("Processed:", res)
        return

    meta = pd.read_csv(CACHE_DIR / "metadata.csv")
    sids = [int(s) for s in meta["subject_id"].tolist()]
    if args.workers <= 1:
        for sid in tqdm(sids, desc="Appending Set A+C"):
            process_subject(sid, overwrite=args.overwrite)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, (sid, args.overwrite)) for sid in sids]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Appending Set A+C ({args.workers}w)"):
                fut.result()
    print("Done appending Set A (eGeMAPS + prosody) and Set C to all cached subjects.")


if __name__ == "__main__":
    main()
