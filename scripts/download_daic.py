#!/usr/bin/env python3
"""
Download only the speech-relevant part of the DAIC-WOZ Kaggle mirror
  https://www.kaggle.com/datasets/saifzaman123445/daicwoz   (version 2)

Keeps per session:  <ID>_AUDIO.wav, <ID>_TRANSCRIPT.csv, <ID>_COVAREP.csv, <ID>_FORMANT.csv
Plus the 4 split label CSVs.  Skips all CLNF_* video features (~115 GB).

Total download: ~16.6 GB across ~756 files.

Layout (default): files are sorted into sub-folders under --out:
    data/raw/DAIC_WOZ/audio/300_AUDIO.wav
                      transcripts/300_TRANSCRIPT.csv
                      covarep/300_COVAREP.csv
                      formant/300_FORMANT.csv
                      splits/train_split_Depression_AVEC2017.csv
Pass --flat to dump everything in one folder instead.

Usage:
    python scripts/download_daic.py                 # -> data/raw/DAIC_WOZ/<type>/
    python scripts/download_daic.py --out D:/daic    # custom target
    python scripts/download_daic.py --workers 6      # parallel downloads
    python scripts/download_daic.py --flat           # no sub-folders

No Kaggle account needed - uses the public dataset endpoint. If you hit rate
limits, drop --workers to 2, or put a kaggle.json token at ~/.kaggle/kaggle.json
and the script will use it automatically.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SLUG = "saifzaman123445/daicwoz"
VERSION = 2
LIST_API = f"https://www.kaggle.com/api/v1/datasets/list/{SLUG}"
DL_API = f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"

# which files to keep, and which sub-folder each lands in
KEEP_SUFFIXES = ("_AUDIO.wav", "_TRANSCRIPT.csv", "_COVAREP.csv", "_FORMANT.csv")
SUBDIR_BY_SUFFIX = {
    "_AUDIO.wav": "audio",
    "_TRANSCRIPT.csv": "transcripts",
    "_COVAREP.csv": "covarep",
    "_FORMANT.csv": "formant",
}
KEEP_EXACT = {
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
    "full_test_split.csv",
}
EXACT_SUBDIR = "splits"


def subdir_for(base: str) -> str:
    if base in KEEP_EXACT:
        return EXACT_SUBDIR
    for suf, sub in SUBDIR_BY_SUFFIX.items():
        if base.endswith(suf):
            return sub
    return ""


def auth_header() -> dict:
    """Use ~/.kaggle/kaggle.json if present (optional)."""
    p = Path.home() / ".kaggle" / "kaggle.json"
    if p.is_file():
        try:
            c = json.loads(p.read_text())
            import base64

            tok = base64.b64encode(f"{c['username']}:{c['key']}".encode()).decode()
            return {"Authorization": f"Basic {tok}"}
        except Exception as e:  # noqa: BLE001
            print(f"  (ignoring unreadable kaggle.json: {e})")
    return {}


HEADERS = {"User-Agent": "Mozilla/5.0"}


def list_files() -> list[tuple[str, int]]:
    """Return [(name, total_bytes), ...] for the whole dataset via the list API."""
    out: list[tuple[str, int]] = []
    token = None
    while True:
        url = LIST_API + (f"?pageToken={token}" if token else "")
        req = urllib.request.Request(url, headers={**HEADERS, **auth_header()})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        for f in data.get("datasetFiles", []):
            out.append((f["name"], int(f.get("totalBytes") or 0)))
        token = data.get("nextPageTokenNullable")
        if not token:
            return out


def wanted(name: str) -> bool:
    base = name.split("/")[-1]
    return base in KEEP_EXACT or base.endswith(KEEP_SUFFIXES)


def download_one(name: str, size: int, out_dir: Path, flat: bool = False, retries: int = 4) -> str:
    base = name.split("/")[-1]
    dest = out_dir / base if flat else out_dir / subdir_for(base) / base
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (size == 0 or dest.stat().st_size == size):
        return f"skip  {dest.name}"
    url = f"{DL_API}?datasetVersionNumber={VERSION}&fileName={name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={**HEADERS, **auth_header()})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            # Kaggle serves text files as a single-entry ZIP (binary/wav come raw).
            with open(tmp, "rb") as fh:
                is_zip = fh.read(4) == b"PK\x03\x04"
            if is_zip:
                import zipfile

                with zipfile.ZipFile(tmp) as z:
                    member = z.namelist()[0]
                    with z.open(member) as src, open(dest, "wb") as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                tmp.unlink(missing_ok=True)
            else:
                tmp.replace(dest)
            if size and dest.stat().st_size != size:
                raise OSError(f"size mismatch {dest.stat().st_size} != {size}")
            return f"ok    {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)"
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            if attempt == retries:
                return f"FAIL  {dest.name}  ({e})"
            time.sleep(2 * attempt)
    return f"FAIL  {dest.name}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/DAIC_WOZ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="download only first N files (testing)")
    ap.add_argument("--flat", action="store_true", help="dump all files in one folder (no subdirs)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Listing dataset files ...")
    all_files = list_files()
    keep = [(n, s) for n, s in all_files if wanted(n)]
    keep.sort()
    if args.limit:
        keep = keep[: args.limit]
    total = sum(s for _, s in keep)
    print(f"  {len(all_files)} files total; downloading {len(keep)}  (~{total/1e9:.1f} GB)")
    print(f"  target: {out_dir.resolve()}\n")

    def path_for(n: str) -> Path:
        b = n.split("/")[-1]
        return out_dir / b if args.flat else out_dir / subdir_for(b) / b

    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download_one, n, s, out_dir, args.flat): n for n, s in keep}
        for fut in cf.as_completed(futs):
            done += 1
            print(f"[{done}/{len(keep)}] {fut.result()}")

    fails = [n for n, s in keep if not path_for(n).exists()]
    if fails:
        print(f"\n{len(fails)} file(s) failed - re-run the script to retry them.")
        return 1
    print("\nAll files present. Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
