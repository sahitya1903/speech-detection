# Speech-Based Depression Detection with Graph Neural Networks

A speaker-independent, speech-centred depression detection pipeline built on the **DAIC-WOZ** clinical interview corpus (189 sessions). The system models each participant's interview as a multi-relational speech graph over **3.0 s acoustic-prosodic windows** and benchmarks classical ML, sequence models (**BiLSTM**), and a pure-PyTorch **Relational Graph Attention Network (RGAT)** designed for **TorchScript** export toward edge deployment.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-DAIC--WOZ%20(189%20sessions)-informational)

---

## Key Highlights

- **Strict Leakage Prevention:** Strips all virtual-agent (*Ellie*) turns using transcript timestamps before windowing, ensuring models never learn interviewer cues.
- **Speaker-Independent 5-Fold CV:** Stratified on `label × gender` across all **189 DAIC-WOZ subjects** (`assert_no_speaker_leakage` in `src/train/cv.py`), with scalers fit strictly inside training folds.
- **3.0 s Windowed Tensor Cache:** Segments participant speech into 3.0 s windows (`1.5 s` hop, capped at `256` nodes/subject) and caches `.pt` graphs per subject for fast CPU training.
- **Multi-Relational RGAT Encoder:** Pure-PyTorch `RelationalGATLayer` (`src/models/encoder_rgat.py`) operating over temporal (`E_seq`) and acoustic-similarity (`E_knn`) edges with no external C-extension dependencies—verified for **TorchScript** export (`results/p2_export_smoke_test.ok`).
- **Comprehensive Evaluation:** Reports **UAR** (Unweighted Average Recall), **Macro-F1**, **Depressed F1**, **ROC-AUC**, 95% bootstrap confidence intervals, feature-group ablations, and **male vs. female fairness gaps**.

---

## System Pipeline

```text
DAIC-WOZ (COVAREP + FORMANT + Audio + Transcripts)
   │
   ├─► Strip Ellie turns (participant speech only)
   ├─► Segment into 3.0 s windows (1.5 s hop, max 256 nodes)
   ├─► Extract Set A (COVAREP, FORMANT, eGeMAPS, Praat CPPS, rhythm/timing)
   ├─► Extract Set C (participant-only linguistic cues, optional ablation)
   └─► Cache per-subject tensors ──► data/cache/DAIC_WOZ/<subject_id>.pt
                                         │
                                         ▼
                        5-Fold Speaker-Independent CV (label × gender)
                          ├── Classical Baselines (SVM, LogReg, Gaussian NB, RF)
                          ├── Sequence Baseline (BiLSTM)
                          └── Multi-Relational RGAT GNN (E_seq + E_knn)
```

### Node Features & Graph Construction
- **Acoustic & Prosodic (Set A):** 100 Hz COVAREP (F0, VUV, glottal-source NAQ/QOQ/H1-H2/peak slope/Rd, MCEP, HMPDM/HMPDD) + FORMANT (F1–F5), openSMILE eGeMAPS, and Praat/Parselmouth prosody & rhythm (CPPS, syllable rate, articulation rate, pause stats, speech-to-pause ratio, F0 slope, words/sec, response latency).
- **Linguistic Ablation (Set C):** Lightweight turn-level features from **participant text only** (pronoun rates, VADER sentiment/emotion rates, absolutist words, fillers, non-verbal tags, verbosity) via `src/data/features_linguistic.py`.
- **Graph Edges (`src/data/build_graph.py`):**
  - `E_seq` (temporal): Connects consecutive speech windows to capture prosodic and rhythmic trajectory across the interview.
  - `E_knn` (feature similarity): Connects acoustically similar windows across time to capture recurring vocal patterns.
  - `E_rec` (optional recurrence): Configurable recurrence edges for graph topology experiments.

---

## Benchmark Results (5-Fold Speaker-Independent CV)

Evaluated across all 189 DAIC-WOZ subjects (`results/end_review/summary_comparison.csv` & `results/end_review/fairness_gender.csv`):

| Model | UAR (Mean ± Std) | 95% CI (UAR) | Macro-F1 | ROC-AUC | Gender UAR Gap (\|F − M\|) |
|---|---:|---:|---:|---:|---:|
| **BiLSTM** | **0.642 ± 0.047** | [0.565, 0.712] | **0.620** | **0.635** | **0.007** |
| **RGAT GNN** | **0.589 ± 0.064** | [0.514, 0.662] | **0.586** | 0.557 | 0.026 |
| Logistic Regression | 0.570 ± 0.095 | [0.493, 0.643] | 0.561 | 0.601 | 0.125 |
| Gaussian NB | 0.562 ± 0.088 | [0.484, 0.640] | 0.542 | 0.576 | 0.017 |
| Random Forest | 0.532 ± 0.124 | [0.467, 0.595] | 0.513 | 0.579 | 0.004 |
| SVM (RBF) | 0.500 ± 0.000 | [0.500, 0.500] | 0.413 | 0.481 | 0.000 |

### Feature-Group Ablation Highlights (`results/end_review/feature_ablation_cv.csv`)
- **All combined features** with Gaussian NB reached **0.592 UAR** (`0.595 ROC-AUC`).
- **Set A + Set C** with Logistic Regression improved UAR from `0.534` (Set A only) to **0.570** (`0.601 ROC-AUC`).
- **eGeMAPS alone** with Gaussian NB achieved **0.564 UAR**, outperforming raw COVAREP (`0.550 UAR`).

---

## Repository Structure

```text
speech-detection/
├── notebooks/
│   ├── 01_mid_review_features_expanded.ipynb   # EDA, statistical tests, and 12-model sklearn benchmark
│   └── 02_end_review_pipeline.ipynb            # End-to-end 5-fold CV training, RGAT, fairness & TorchScript export
├── scripts/
│   ├── download_daic.py                        # DAIC-WOZ download helper
│   ├── update_tracker.py                       # Automated progress & filesystem verification
│   └── build_end_review_ppt.py                 # Review presentation generator
├── src/
│   ├── data/
│   │   ├── preprocess.py                       # Ellie-turn stripping, 3.0s windowing, .pt cache builder
│   │   ├── features_prosody.py                 # Praat CPPS, rhythm, pause & timing feature extraction
│   │   ├── features_linguistic.py              # Participant-only Set C linguistic features
│   │   └── build_graph.py                      # Multi-relational graph builder (E_seq, E_knn, E_rec)
│   ├── baselines/
│   │   ├── svm_rf.py                           # Classical ML baselines (SVM, LogReg, NB, RF)
│   │   └── bilstm.py                           # Sequence baseline (BiLSTM)
│   ├── models/
│   │   ├── encoder_rgat.py                     # Pure-PyTorch TorchScript-compatible RGATConv encoder
│   │   └── classifier.py                       # Graph readout (mean/max/attention) + MLP classifier
│   └── train/
│       └── cv.py                               # 5-fold speaker-independent CV harness & leakage assertions
├── results/
│   ├── mid_review/                             # Mid-review plots, RFE sweeps, AVEC 2016 comparisons
│   ├── end_review/                             # CV metrics, OOF predictions, checkpoints, fairness & ablation CSVs
│   └── p2_export_smoke_test.ok                 # TorchScript export verification marker
├── requirements.txt
├── IMPLEMENTATION_PLAN.md
├── TRACKER.md
└── LICENSE
```

---

## Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/sahitya1903/speech-detection.git
cd speech-detection
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Access (DAIC-WOZ)
DAIC-WOZ requires a research license agreement through USC ICT. Once approved, place the dataset under `DAIC_WOZ/` (or use `scripts/download_daic.py` with your credentials).

### 3. Build the Windowed Tensor Cache
```bash
python -m src.data.preprocess
```
This strips interviewer turns, segments participant speech into 3.0 s windows, and writes per-subject `.pt` files to `data/cache/DAIC_WOZ/`.

### 4. Run Experiments
Open and run `notebooks/02_end_review_pipeline.ipynb` (or `01_mid_review_features_expanded.ipynb` for the initial feature-selection study) to reproduce the 5-fold CV benchmarks, ablation tables, fairness slices, and TorchScript export check.

---

## Roadmap

- **Architecture Enhancements:** Hierarchical graph pooling (`SAGPool`/`ASAP`), anti-oversmoothing controls (`DropEdge`, jumping knowledge), and an adversarial gender-debiasing head.
- **Cross-Corpus Generalization:** Zero-shot evaluation on **E-DAIC** (and optionally DEPAC) + segment-importance interpretability timelines.
- **Edge Deployment:** INT8 post-training quantization and latency/memory benchmarking on **Raspberry Pi** and **Jetson Nano** using the TorchScript-exported RGAT backbone.

---

## License

This project is licensed under the [MIT License](LICENSE).
