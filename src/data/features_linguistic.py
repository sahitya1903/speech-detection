"""
P1 Set C: Lightweight Linguistic Features (Participant Turns Only — Zero Ellie Leakage).

Computes 8 interpretable linguistic features per participant turn from
DAIC_WOZ/transcripts/<sid>_TRANSCRIPT.csv and maps them onto each 3 s segment
using the cached `turn_ids` tensor:
  1. first_person_sg_rate  ("i", "me", "my", "myself", ...)
  2. first_person_pl_rate  ("we", "us", "our", "ourselves", ...)
  3. neg_emotion_rate      (VADER negative lexicon / fallback)
  4. pos_emotion_rate      (VADER positive lexicon / fallback)
  5. absolutist_rate       (Al-Mosaiwi & Johnstone 2018)
  6. filler_rate           ("um", "uh", "mm", "hmm", ...)
  7. nonverbal_rate        (<laughter>, <sigh>, etc.)
  8. turn_word_count       (verbosity per turn)
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import torch

from src.data.preprocess import DAIC_DIR, load_participant_turns

FIRST_SG = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}
FIRST_PL = {"we", "us", "our", "ours", "ourselves", "we're", "we've", "we'd", "we'll"}
ABSOLUTIST = {
    "always", "never", "nothing", "everything", "completely", "totally",
    "entirely", "absolutely", "definitely", "constantly", "forever",
    "everyone", "nobody", "none", "all", "every", "must",
}
FILLERS = {"um", "uh", "mm", "hmm", "uhh", "umm", "er", "erm", "mhm"}

# Lightweight fallback emotion lexicons if vaderSentiment is unavailable
_FALLBACK_NEG = {
    "sad", "depressed", "lonely", "anxious", "hopeless", "worthless", "tired",
    "bad", "worse", "worst", "hate", "pain", "hurt", "cry", "crying", "stress",
    "stressed", "afraid", "scared", "angry", "upset", "hard", "difficult",
    "fail", "failed", "failure", "empty", "numb", "guilty", "sorry", "terrible",
}
_FALLBACK_POS = {
    "good", "great", "happy", "glad", "love", "loved", "enjoy", "enjoyed",
    "nice", "well", "better", "best", "fine", "fun", "hope", "hopeful",
    "excited", "proud", "calm", "relaxed", "grateful", "thankful", "amazing",
}

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
    _VADER_LEX = _VADER.lexicon
except Exception:
    _VADER_LEX = None

_TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _turn_features(text: str) -> np.ndarray:
    """Compute 8-dim linguistic vector for one participant turn."""
    raw = str(text).lower()
    nonverbal_hits = len(_TAG_RE.findall(raw))
    cleaned = _TAG_RE.sub(" ", raw)
    words = _WORD_RE.findall(cleaned)
    n_words = len(words)
    denom = max(1, n_words)

    sg_cnt = sum(1 for w in words if w in FIRST_SG)
    pl_cnt = sum(1 for w in words if w in FIRST_PL)
    abs_cnt = sum(1 for w in words if w in ABSOLUTIST)
    fil_cnt = sum(1 for w in words if w in FILLERS)

    if _VADER_LEX is not None:
        neg_cnt = sum(1 for w in words if _VADER_LEX.get(w, 0.0) < -0.5)
        pos_cnt = sum(1 for w in words if _VADER_LEX.get(w, 0.0) > 0.5)
    else:
        neg_cnt = sum(1 for w in words if w in _FALLBACK_NEG)
        pos_cnt = sum(1 for w in words if w in _FALLBACK_POS)

    return np.array(
        [
            sg_cnt / denom,
            pl_cnt / denom,
            neg_cnt / denom,
            pos_cnt / denom,
            abs_cnt / denom,
            fil_cnt / denom,
            nonverbal_hits / denom,
            float(n_words),
        ],
        dtype=np.float32,
    )


def extract_set_c_for_subject(
    subject_id: int,
    turn_ids: torch.Tensor,
    daic_dir: Path = DAIC_DIR,
) -> torch.Tensor:
    """Return [n_nodes, 8] Set C tensor aligned to each segment's turn_id."""
    turns = load_participant_turns(daic_dir / "transcripts" / f"{subject_id}_TRANSCRIPT.csv")
    turn_vecs = [_turn_features(val) for val in turns["value"].tolist()]
    if not turn_vecs:
        return torch.zeros((len(turn_ids), 8), dtype=torch.float32)

    mat = np.stack(turn_vecs, axis=0)
    idx = np.clip(turn_ids.numpy(), 0, len(mat) - 1)
    return torch.from_numpy(mat[idx])
