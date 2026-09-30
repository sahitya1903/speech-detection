#!/usr/bin/env python3
"""
Refresh TRACKER.md against the actual repo state.

- Every checkbox tagged `<!--auto:KEY-->` gets its checked state overwritten
  based on the corresponding check in AUTO_CHECKS below (file exists, session
  counts match, etc.). These boxes reflect the filesystem, not opinion.
- Every untagged checkbox is left exactly as-is (manual judgment calls).
- The progress table between PROGRESS_TABLE_START/END markers is regenerated
  from checkbox counts per '## Pn' section.

Run after any implementation step:  python scripts/update_tracker.py
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "TRACKER.md"

DAIC = ROOT / "DAIC_WOZ"
SRC = ROOT / "src"
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Known-excluded official DAIC-WOZ session IDs (technical issues, per USC).
EXCLUDED_IDS = {342, 394, 398, 460}
EXPECTED_SESSIONS = set(range(300, 493)) - EXCLUDED_IDS  # 189 IDs


def _count_suffix(folder: Path, suffix: str) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.name.endswith(suffix))


def _exists_nonempty(path: Path, min_lines: int = 1) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return len(text.splitlines()) >= min_lines


def check_daic_downloaded() -> bool:
    audio_ids = set()
    audio_dir = DAIC / "audio"
    if audio_dir.is_dir():
        for p in audio_dir.iterdir():
            m = re.match(r"(\d+)_AUDIO\.wav$", p.name)
            if m:
                audio_ids.add(int(m.group(1)))
    if audio_ids != EXPECTED_SESSIONS:
        return False
    for sub, suffix in (("transcripts", "_TRANSCRIPT.csv"), ("covarep", "_COVAREP.csv"), ("formant", "_FORMANT.csv")):
        if _count_suffix(DAIC / sub, suffix) != len(EXPECTED_SESSIONS):
            return False
    return (DAIC / "splits").is_dir()


def check_cache_built() -> bool:
    cache_dir = DATA / "cache" / "DAIC_WOZ"
    if not cache_dir.is_dir():
        return False
    n = sum(1 for p in cache_dir.glob("*.pt"))
    return n >= len(EXPECTED_SESSIONS)


def check_build_graph_stub() -> bool:
    return (SRC / "data" / "build_graph.py").is_file()


def check_graph_edges_impl() -> bool:
    """Heuristic: same file as the P0 stub, but 'implemented' means it actually
    references the edge types from the plan and isn't just a stub."""
    f = SRC / "data" / "build_graph.py"
    if not f.is_file():
        return False
    text = f.read_text(encoding="utf-8", errors="ignore").lower()
    hits = sum(kw in text for kw in ("e_seq", "e_knn", "e_rec", "knn", "temporal"))
    return hits >= 2 and len(text.splitlines()) >= 30


AUTO_CHECKS = {
    "daic_downloaded": check_daic_downloaded,
    "preprocess_written": lambda: _exists_nonempty(SRC / "data" / "preprocess.py", 10),
    "build_graph_stub": check_build_graph_stub,
    "cache_built": check_cache_built,
    "eda_notebook": lambda: any((ROOT / "notebooks").glob("*.ipynb")) if (ROOT / "notebooks").is_dir() else False,
    "cv_harness": lambda: _exists_nonempty(SRC / "train" / "cv.py", 10),
    "svm_rf_baseline": lambda: _exists_nonempty(SRC / "baselines" / "svm_rf.py", 5),
    "bilstm_baseline": lambda: _exists_nonempty(SRC / "baselines" / "bilstm.py", 5),
    "wav2vec_probe_baseline": lambda: _exists_nonempty(SRC / "baselines" / "wav2vec_probe.py", 5),
    "graph_edges_impl": check_graph_edges_impl,
    "rgat_encoder": lambda: _exists_nonempty(SRC / "models" / "encoder_rgat.py", 10),
    "classifier_assembly": lambda: _exists_nonempty(SRC / "models" / "classifier.py", 5),
    "onnx_export_test": lambda: (RESULTS / "p2_export_smoke_test.ok").is_file(),
    "pooling_impl": lambda: _exists_nonempty(SRC / "models" / "pooling.py", 5),
    "contrastive_pretrain_impl": lambda: _exists_nonempty(SRC / "models" / "pretrain_graphcl.py", 5),
    "adversary_impl": lambda: _exists_nonempty(SRC / "models" / "adversary.py", 5),
    "ablation_table": lambda: (RESULTS / "ablation_table.csv").is_file(),
    "features_ssl_impl": lambda: _exists_nonempty(SRC / "data" / "features_ssl.py", 5),
    "edaic_downloaded": lambda: (ROOT / "E_DAIC").is_dir() and any((ROOT / "E_DAIC").iterdir()),
    "depac_downloaded": lambda: (ROOT / "DEPAC").is_dir() and any((ROOT / "DEPAC").iterdir()),
    "cross_corpus_impl": lambda: _exists_nonempty(SRC / "eval" / "cross_corpus.py", 5),
    "interpret_impl": lambda: _exists_nonempty(SRC / "eval" / "interpret.py", 5),
    "ib_subgraph_impl": lambda: _exists_nonempty(SRC / "models" / "ib_subgraph.py", 5),
    "clnf_downloaded": lambda: any(DAIC.rglob("*CLNF*")),
    "edge_export": lambda: any((ROOT / "models_edge").glob("*.onnx")) or any((ROOT / "models_edge").glob("*.pt")) if (ROOT / "models_edge").is_dir() else False,
    "benchmark_rpi": lambda: (RESULTS / "benchmark_rpi.json").is_file(),
    "benchmark_jetson": lambda: (RESULTS / "benchmark_jetson.json").is_file(),
}


def apply_auto_checks(text: str) -> str:
    pattern = re.compile(r"- \[( |x)\]([^\n]*?)<!--auto:(\w+)-->")

    def repl(m: re.Match) -> str:
        _checked, rest, key = m.group(1), m.group(2), m.group(3)
        fn = AUTO_CHECKS.get(key)
        ok = False
        if fn is not None:
            try:
                ok = bool(fn())
            except Exception:
                ok = False
        box = "x" if ok else " "
        return f"- [{box}]{rest}<!--auto:{key}-->"

    return pattern.sub(repl, text)


def regenerate_progress_table(text: str) -> str:
    rows = []
    for m in re.finditer(r"^## (P\d+[^\n]*)\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL):
        title, body = m.group(1), m.group(2)
        boxes = re.findall(r"- \[( |x)\]", body)
        total = len(boxes)
        done = sum(1 for b in boxes if b == "x")
        pct = f"{(100 * done / total):.0f}%" if total else "—"
        rows.append(f"| {title} | {done} | {total} | {pct} |")
    table = "| Phase | Done | Total | % |\n|---|---|---|---|\n" + "\n".join(rows)
    return re.sub(
        r"(<!-- PROGRESS_TABLE_START -->\n).*?(\n<!-- PROGRESS_TABLE_END -->)",
        lambda m: m.group(1) + table + m.group(2),
        text,
        flags=re.DOTALL,
    )


def update_timestamp(text: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return re.sub(r"\*Last auto-check: .*?\*", f"*Last auto-check: {ts}*", text)


def main() -> None:
    text = TRACKER.read_text(encoding="utf-8")
    text = apply_auto_checks(text)
    text = regenerate_progress_table(text)
    text = update_timestamp(text)
    TRACKER.write_text(text, encoding="utf-8")
    print(f"Updated {TRACKER.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
