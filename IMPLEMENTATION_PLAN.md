# Speech-Based Depression Detection with Graph Neural Networks — Implementation Plan

## 0. Scope & assumptions

Locked decisions (from project discussion):

| Dimension | Decision |
|---|---|
| Modality | **Speech-centred v1**: the acoustic signal is the core input. A small, switchable **linguistic feature group (Set C, §3)** from the participant's own transcript text is added as an ablation, following the guide's 2026-10-01 feature guidance. No deep text models, no video — see §14 for the planned optional video extension. |
| Task | **Binary classification**: depressed vs. non-depressed |
| Core model family | **Graph Neural Network** on speech-derived graphs |
| Compute budget | 1 laptop: Core Ultra 5 225H, Intel Arc iGPU (no CUDA), 16 GB RAM |
| Eventual deployment target | Raspberry Pi / Jetson Nano, **after** the full software project (through P6) is done — see §15 |

Design consequence of the compute budget: **all heavy feature extraction is done once, offline, and cached to disk as tensors.** The GNN itself operates on those cached features and is tiny (a few hundred thousand parameters, ~200–1000 graphs), so it trains on CPU in minutes. This is the single most important architectural constraint and it shapes every choice below.

**Roadmap note (why §14/§15 exist but aren't v1):** video (§14) and edge deployment (§15) are confirmed future phases, not hypothetical stretch ideas — but neither should change how you build P0–P6. The only thing they change *now* is two design habits, both cheap to follow from the start: (1) keep the node/segment **feature encoder** (openSMILE, wav2vec2) and the **graph backbone** (RGAT/pooling) as separate modules so a video encoder can plug into the same backbone later instead of forcing a rewrite; (2) treat "does this export to ONNX/TorchScript with a fixed max-node-count" as a running constraint on the GNN backbone, checked once cheaply in P2 (see the added P2 exit criterion in §10), so P5/P6 architecture choices don't quietly paint you into a corner for §15.

Grounding papers:
- Leal, Ntalampiras, Sassi — *Speech-Based Depression Assessment: A Comprehensive Survey* (IEEE TAFFC 2025). Referred to below as **SDR-survey**.
- Macky Aleagha, Zohari, Chehreghani — *AI Models for Depressive Disorder Detection and Diagnosis: A Review* (arXiv 2508.12022). Referred to below as **AI-review**.

---

## 0.5 This-semester timeline, team split & presentation strategy

This is officially a **two-semester** capstone project, but per the mentor's guidance the entire technical scope below — through P3 (core increments) and P8 (edge deployment) — is being completed within this single semester. Institute rules still require showing incremental progress across both semesters, so there are two tracks from here on. This section is the technical-plan-side summary; the "Study & Execution Plan" PDF (generated 2026-09-14, updated iteratively as constraints were discovered) is the authoritative day-to-day calendar — keep the two in sync rather than letting them drift.

**Real (internal) deadline:** P0, P1, P2, and P3-core (6.1 hierarchical pooling, 6.3 anti-oversmoothing, 6.4 adversarial gender head) plus P8 are engineering-complete by **~5 Nov 2026** (a 2-day buffer before the actual End-Review presentation slot). P4, P5, and P7 are a conditional **stretch tier** (below) — attempted only if ahead of schedule, never at the cost of the 5 Nov deadline. 6.2 (contrastive pretraining) is likewise attempted only if P3-core finishes early.

**Official presentation split (deliberately rebalanced — an earlier version of this plan held back only P8 for next semester, which was correctly flagged as a lopsided "did everything except demo it on a Pi" story):**
- *This semester's End-Review* — official window 7–26 Nov 2026, but the real presentation slot is **~7–9 Nov** since End-Semester exams (Lab 10–12 Nov, Theory 13–26 Nov) occupy the rest of that window. The deck presents only **P0 + P1 + a basic/flat P2 GNN** (RGAT + flat readout vs. baselines) as complete. P3's full increments, the ablation table, statistical testing, fairness/interpretability analysis, and P8 edge deployment are all *already finished* by this point internally but deliberately left out of the deck's Results section, appearing only under "Conclusions & Future Scope."
- *Next semester's final review* — officially reveals P3's full architecture work + P8 edge deployment + the final paper, giving that semester a substantial, legitimate "new work" story.
- P8 is demoed privately to the mentor this semester (~5 Nov) but withheld from the judging panel until next semester.

**Team split** (two students; both are individually assessed on contribution/understanding at each review — splitting tasks is for throughput, not for either student to skip understanding their assigned half):
- **Student A ("you")**: P0 data pipeline, P2 graph construction + RGAT encoder, P3 architecture increments.
- **Student B ("partner")**: P1 baselines + the speaker-independent CV harness; pre-stages the Raspberry Pi/Jetson environment ahead of the P8 sprint.
- **P8 edge deployment**: joint work by both students, once the P3 hero model is frozen — this is the part demoed to the mentor and needs both students individually fluent in it.

**The paper (P6) is written only after the whole committed scope is finished** — and because of the End-Sem exam blackout (10–26 Nov), that means **27 Nov 2026 or later**, not immediately after the End-Review. This reverses the general phase ordering in §10 below, where P6 ("software project complete") gates P7/P8 — in this semester's compressed execution, **P8 happens before the write-up**, because the write-up is explicitly deferred past the exam blackout while the engineering work is not. Treat the §10 ordering as the general/idealized version of this plan and this section as the override for this semester's actual execution.

**Stretch tier** (attempt only if the 5 Nov deadline is beaten with room to spare — either inside the Oct–Nov build window, or more realistically in the lower-pressure gap after 27 Nov before next semester's final review. Attempt in this order; stop whenever the slack runs out; fold whatever finishes into the paper as real results, not "future work" placeholders):
1. **P4** — E-DAIC cross-corpus generalization. Highest value: both grounding surveys name cross-corpus generalization the field's central open problem. Moderate cost (download E-DAIC, zero-shot eval on the frozen hero model, optional light fine-tune).
2. **P5** — 6.5 IB subgraph selection, then 6.6 two-level population graph, then 6.7 acoustic-event tokenization edges, evaluated one at a time — kept only if each beats the hero model, per the plan's own rule in §6.
3. **P7** — video/CLNF fusion. Lowest stretch priority, highest cost: ~75–135 GB of extra CLNF/HOG download (check free disk space first) plus a second graph branch/fusion architecture. Realistically only reachable in the 27 Nov+ window.

---

## 1. Dataset choice

Two hard requirements: **English-language speech**, and **small / simple enough for a 16 GB laptop with no CUDA GPU**. English rules out EATD, MODMA, CMDC, CONVERGE (all Mandarin) and the Androids Corpus (Italian). Realistic English options:

| Dataset | Size | Speech type | Label | Fit for this project |
|---|---|---|---|---|
| **DAIC-WOZ** | 189 sessions, 7–33 min; **provided feature CSVs ≈ 1–3 GB**, raw audio ≈ 15 GB (no video) | Semi-structured clinical interview (Ellie virtual agent), spontaneous | PHQ-8; depressed if ≥ 10 | **Primary.** The field benchmark → direct comparability with almost every SDR/GNN paper in both surveys (AVEC 2016/17). Laptop-safe if you start from the shipped COVAREP + FORMANT features and add raw-audio eGeMAPS / wav2vec2 later. |
| **E-DAIC (Extended DAIC-WOZ)** | 275 participants; feature package similar scale | Semi-clinical interview, fully automated agent | PHQ-8 | **Secondary / external validation.** Same family as DAIC-WOZ but different collection protocol → a fair generalization test. |
| **DEPAC** | 571 participants, ~2 674 recordings, short **structured tasks** (phonemic + semantic fluency, picture description, prompted narrative, reading) | Structured elicitation | PHQ-9, GAD-7 | **Strong alternative primary.** Shorter clips than DAIC → *lighter* to featurize; large N; ships an acoustic feature set. English (Canadian). Available on request (Novikova et al., CLPsych 2022 — SDR-survey [30]). |
| D-Vlog | 961 YouTube vlogs (~1–10 min) | "In the wild" monologue | Weak self-disclosure labels | **Optional robustness / in-the-wild check only.** Noisy audio, weak labels — not for the main result (AAAI 2022). |

**Plan:**
1. Build and tune everything on **DAIC-WOZ**, starting from the **pre-extracted acoustic feature CSVs** (COVAREP: F0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, peak slope, Rd, creak, MCEP, HMPDM, HMPDD; plus FORMANT F1–F5). This path needs **no raw-audio processing at all** and is the lightest possible on your hardware. **Note (verified 2026-10-04 against the AVEC 2017 documentation): DAIC-WOZ ships only COVAREP + FORMANT, not openSMILE/eGeMAPS.** eGeMAPS must be extracted from the raw audio with the `opensmile` package (minutes, one-time).
2. Add the **raw-audio wav2vec2/HuBERT (English) node-feature stream** as an ablation once the pipeline works — extraction is a one-time cached pass.
3. Freeze the pipeline, then run it unchanged on **E-DAIC** (and optionally DEPAC) as a cross-protocol generalization test — the SDR-survey repeatedly flags cross-corpus generalization as *the* open problem. Do not touch the held-out corpus until Phase 4.

If DAIC-WOZ's 20-minute interviews still feel heavy, **swap the primary to DEPAC** — same plan, shorter clips, more subjects, still English.

Access notes:
- DAIC-WOZ / E-DAIC: USC ICT license agreement (free for research, ~1 day turnaround). Download the **audio + transcripts + pre-extracted features** package; **skip the video** (~130 GB).
- DEPAC: request from the authors (WinterLight / Novikova group).
- Use participant speech only — strip all "Ellie" / virtual-agent turns (transcripts give turn timestamps).

---

## 2. Data pipeline

```
provided feature CSV ─┐
                      ├─► align to participant turns ──► segment into windows ──► node features ──► cache (.pt)
raw wav ──► 16 kHz mono ──► VAD / silence trim ──► participant-only concat ──┘   (raw-audio path, optional)
```

Details:

1. **Feature-CSV path (default):** the DAIC-WOZ package ships frame-level COVAREP + FORMANT features (100 Hz); openSMILE/eGeMAPS ships only with E-DAIC. Load these directly — **no signal processing needed**. Filter to participant-speech frames using transcript turn timestamps.
2. **Raw-audio path (optional upgrade):** resample to 16 kHz mono; **voice-activity detection** (Silero VAD) to drop leading/trailing silence and long gaps. Keep pause *structure* inside speech — pause rate and speech rate are established depression markers (SDR-survey §IV-B), so cap silences at ~1.5 s rather than removing them.
3. **Per subject**, keep participant turns only — strip every "Ellie" / virtual-agent turn (transcripts give the boundaries).
4. **Segment** into windows of **W = 3 s, hop = 1.5 s** (50% overlap). Each window becomes one graph node. A DAIC session → roughly 100–400 participant-speech nodes; sub-sample to a cap (e.g. 256) if needed for memory.
   - Rationale: short enough to be locally stationary, long enough for stable prosody; overlap gives temporal continuity for edges.
5. **Cache** everything under `data/cache/<dataset>/<subject_id>.pt` containing: node feature matrix `X` `[n_nodes, d]`, segment timestamps, the turn id of each segment (needed to copy turn-level features onto segments, §3), subject label, gender, subject id. Never re-run extraction in the training loop.

**Speaker-independent splits from day one.** Split by *subject*, never by segment. Use 5-fold subject-stratified CV (stratify on label × gender). This is non-negotiable — the SDR-survey shows speaker-dependent numbers (95 %+) are dramatically optimistic vs. speaker-independent.

Class imbalance (DAIC-WOZ is roughly 1:3 depressed:control, and the official train/dev/test split is itself imbalanced): handle with (a) weighted BCE loss, (b) segment-level augmentation of the minority class, (c) report F1 / AUC / UAR, not accuracy (both surveys warn accuracy is misleading here). Prefer speaker-independent CV over the official split, or report both.

---

## 3. Node features (per 3 s segment)

Three feature sets, all precomputed. Start with A, add C as an ablation in P1, and add B as an ablation in P4.

**Guide's guidance (2026-10-01):** combine complementary feature groups (prosody, voice quality, spectral, temporal behaviour, linguistic content) rather than relying on one. The table below maps every feature the guide named to where it comes from in this plan. Use it as-is for the review slides.

| Guide's group | Feature | Source | Set | Scale |
|---|---|---|---|---|
| Prosodic | F0 mean, pitch variability (std, percentile range) | eGeMAPS (F0 in semitones) | A | segment |
| Prosodic | Pitch contour slope (rising / falling) | eGeMAPS slopes + explicit F0 slope (Parselmouth) | A | segment |
| Prosodic | Energy / intensity (mean, variability, peaks/sec) | eGeMAPS loudness | A | segment |
| Prosodic | Speech rate: syllables/sec | Parselmouth syllable-nuclei count | A | turn → segment |
| Prosodic | Speech rate: words/sec | Transcript word count ÷ turn duration | A | turn → segment |
| Prosodic | Pauses: count, mean duration, % silence | Parselmouth intensity-based silence detection; eGeMAPS unvoiced-segment stats | A | turn → segment |
| Prosodic (rhythm) | Articulation rate (syllables ÷ *speaking* time, pauses excluded) | Parselmouth | A | turn → segment |
| Prosodic (rhythm) | Speech-to-pause ratio | Derived from the pause stats above | A | turn → segment |
| Prosodic (timing) | Response latency (end of Ellie's question → participant starts) | Transcript timestamps | A | turn → segment |
| Voice quality | Jitter, shimmer | eGeMAPS (COVAREP has neither; Parselmouth as cross-check) | A | segment |
| Voice quality | Glottal-source measures: NAQ, QOQ, H1-H2, peak slope, Rd | Shipped COVAREP CSVs | A | segment |
| Voice quality | Harmonicity (HNR) | eGeMAPS | A | segment |
| Voice quality | Cepstral Peak Prominence (CPP, smoothed; mean + std) | Parselmouth (Praat "Get CPPS"). **Not in eGeMAPS or COVAREP; added explicitly.** | A | segment |
| Spectral | MFCC 1–4, spectral slope, alpha ratio, Hammarberg index, formants F1–F3 | eGeMAPS; shipped FORMANT CSVs | A | segment |
| Temporal behaviour | How the above evolve across the interview | Graph structure: temporal + similarity edges (§4) | — | graph |
| Linguistic | See Set C below | Participant transcript text | C | turn → segment |

"turn → segment" means the feature is computed once per participant turn and copied onto every 3 s segment inside that turn. Rate and pause measures are too noisy inside a single 3 s window, because a window often catches only part of a pause.

**Set A: handcrafted acoustic + timing (CPU, fast, interpretable):**
- **eGeMAPS v02** functionals via openSMILE (`opensmile` Python package) → 88-dim. Covers prosodic (F0, loudness, jitter, shimmer, HNR), spectral, and cepstral (MFCC 1–4) features. This is the SDR-survey's reference feature set.
- Additional **prosody, rhythm and voice-quality** features, each listed in the table above → ~11 dims:
  - syllable rate, articulation rate, speech-to-pause ratio
  - pause count, mean pause length, % silence
  - F0 slope
  - CPP mean and std
  - words/sec, response latency
- Total ≈ 99-dim. z-score normalize using **training-fold statistics only**.
- The additional features need the raw audio (already downloaded), not just the shipped CSVs. Build the cache from the CSVs first so P0 isn't blocked, then append these columns.

**Set C: lightweight linguistic features (participant text only; added as an ablation):**
- Computed per participant turn from the DAIC-WOZ transcripts, then copied onto that turn's segments. ~8 dims:
  - first-person singular pronoun rate ("i", "me", "my", "myself")
  - first-person plural rate ("we", "us", "our")
  - negative-emotion word rate and positive-emotion word rate, from a free lexicon such as the NRC Emotion Lexicon or VADER; LIWC is commercial, so avoid it
  - absolutist word rate ("always", "never", "nothing", "completely", …), following Al-Mosaiwi & Johnstone 2018
  - filler rate: "um", "uh", "mm", "hmm" are transcribed explicitly
  - non-verbal event rate: `<laughter>`, `<sigh>` and similar tags are transcribed
  - mean words per turn (verbosity)
- **Leakage rule (strict):** use **participant turns only**. Ellie's prompts and follow-up questions depend on the participant's state and leak the label, so never compute any feature from Ellie's text.
- Kept switchable (`use_linguistic: true/false` in config) so every result can be reported with and without it.
- **Not part of the edge profile (§15):** DAIC-WOZ transcripts are human-written, and on-device use would need speech recognition, which is out of scope for P8.

**Set B — self-supervised speech embedding (one-time CPU pass, then frozen):**
- English `wav2vec2-base` or `HuBERT-base` via HuggingFace. Mean+std pool the last hidden layer over each 3 s window → 1536-dim, then PCA to 128-dim (fit on training folds) to keep the GNN small.
- AI-review + SDR-survey both report SSL speech embeddings (wav2vec2, DEPA) beat handcrafted features; but they cost more. Keep frozen — **do not fine-tune wav2vec2** on this hardware.

**Final node vector:** `[Set A | Set C | Set B_pca]` ≈ 235-dim (configurable). The A-only path must work standalone. For DAIC-WOZ, eGeMAPS is extracted from raw audio. The fastest start is a COVAREP + FORMANT segment vector (~79-dim) from the shipped CSVs, with eGeMAPS and the Set A extras appended afterwards.

Extraction cost estimate: COVAREP/FORMANT from the shipped CSVs ≈ seconds; eGeMAPS from raw audio ≈ minutes; Set A extras (CPP, rhythm, pauses) from raw audio ≈ minutes; Set C from transcripts ≈ seconds; Set B on CPU for the full DAIC audio ≈ several hours once, then cached forever.

---

## 4. Graph construction (the core design choice)

Both surveys note there is **no consensus on how to build the graph** — this is exactly where the architectural contribution lives. Build a **multi-relational (heterogeneous-edge) graph per subject** with three edge types, then let the model learn how much to trust each:

1. **Temporal edges** `E_seq`: node *i* ↔ *i*+1 (and optionally *i*±2). Encodes rhythm, speech rate, pause dynamics — the prosodic signature of psychomotor retardation.
2. **Acoustic-similarity edges** `E_knn`: k-NN (k = 5) in node-feature space (cosine). Connects prosodically similar moments regardless of when they occur — lets the model pool "all the flat-affect segments" together. Mirrors the *inter-audio* graph in the two-stage audio GNN (AI-review [62]).
3. **Self-similarity / recurrence edges** `E_rec` (optional): edges where the recurrence-plot distance between segments is below a threshold — captures repetitive prosodic patterns.

Store as a single graph with an `edge_type` attribute → consumed by an **R-GCN / RGAT** layer, or as a **multi-view GNN** (one GNN branch per edge type, then fuse). Start with relational GAT.

**Learnable adjacency refinement:** after the first layer, add a lightweight **graph generator** (MLP on node-pair embeddings → sigmoid → sparsified) that produces a subject-specific learned adjacency, blended with the constructed one:
`A_final = α · A_constructed + (1−α) · A_learned`, α learnable.
This directly answers the "which graph construction is right" contradiction — the model picks.

---

## 5. Baseline model (Phase 2 deliverable)

Keep it deliberately small and boring so later gains are attributable.

```
node features X  [n, d]
   │
   ├─ Linear(d → 128) + ReLU + Dropout(0.3)
   │
GNN block × 2:   RGATConv(128 → 128, heads=4, edge_types=3)
                 + residual + LayerNorm + ReLU
   │
Readout:  concat( mean-pool , max-pool , attention-pool )   → [384]
   │
MLP(384 → 64 → 1)  →  sigmoid  →  P(depressed)
```

- Params ≈ 200–400 K. Trains on CPU in < 5 min/fold.
- Loss: weighted BCE.
- Optimizer: AdamW, lr 1e-3, cosine decay, 100 epochs, early-stop on val UAR.
- **Reference baselines to beat** (implement at least the first two): (a) segment-mean eGeMAPS → SVM/RF (the classic SDR pipeline, and the AVEC 2016 DAIC baseline family), (b) BiLSTM / GRU over the segment-feature sequence, (c) mean-pooled wav2vec2 → linear probe.

Target: match or beat the BiLSTM at equal features. If the plain GNN does not beat the BiLSTM, the graph structure is not helping — fix construction before adding complexity.

---

## 6. Architectural improvements ("improvisation") — staged

Each item is an independent, ablatable increment. Add one at a time; keep it only if it improves speaker-independent val UAR with an honest CV. Ordered by expected value / effort.

### 6.1 Hierarchical graph pooling with interpretability  ⭐ high priority
Replace flat readout with **SAGPool / ASAP** hierarchical pooling: segment nodes → merged into "prosodic regions" → graph vector. Mirrors the EEG SGP-SL model (AI-review [8]) and the local↔global audio GNN (AI-review [62]).
- Benefit 1: better inductive bias for variable-length recordings.
- Benefit 2: **interpretability for free** — the pooling scores say *which 3 s segments drove the decision*. Both surveys make interpretability a hard requirement for clinical credibility (SDR-survey §V, AI-review §7.2). Render these as a timeline heatmap over the waveform.

### 6.2 Self-supervised graph pre-training (contrastive)  ⭐ high priority for small data
Two augmented views of each subject-graph, maximize agreement (GraphCL / dual-branch, as in AI-review [75] DSGNN and [44] NCGL).
- Graph augmentations: node-feature masking, edge dropping (DropEdge), temporal sub-graph crop, **audio-level** SpecAugment / speed-perturb ±10 % / additive noise regenerating node features.
- Pretrain the encoder unsupervised on *all* subjects (labels not needed), then fine-tune the classifier head on labeled folds.
- This is the highest-leverage move against the #1 problem in both surveys: tiny datasets → overfitting. Pretraining can also later absorb unlabeled speech from other corpora.

### 6.3 Anti-oversmoothing / depth controls
Small graphs + deep GNN = oversmoothing (AI-review [40] APO-GCN explicitly targets this).
- Use **GIN** or **APPNP**-style propagation, jumping-knowledge (concat layer outputs), residuals, DropEdge.
- Keep to 2–3 message-passing steps; verify Dirichlet energy doesn't collapse.

### 6.4 Speaker / gender adversarial disentanglement  ⭐ addresses fairness + generalization
Add an auxiliary head predicting **gender** (and, if available, speaker id) from the graph embedding, connected through a **gradient-reversal layer**.
- Pushes the embedding to be depression-informative but gender/speaker-invariant.
- Directly targets SDR-survey KI3/KI7 (bias toward subpopulations; speaker-independent representations) and AI-review §7.4 fairness. Cheap, well-established (non-uniform speaker disentanglement, SDR-survey [31]).
- Report gender-stratified TPR/FPR gap before vs. after.

### 6.5 Information-bottleneck subgraph selection
BrainIB-style (AI-review [86]): learn to sample the **minimal subgraph** sufficient for prediction, with a mutual-information regularizer.
- Improves generalization on small data and yields a compact, clinician-readable "evidence subgraph."
- Higher implementation cost; schedule after 6.1–6.4.

### 6.6 Two-level (population) graph — optional, transductive
Second graph where each **node is a whole subject**, edges by acoustic-embedding / demographic similarity; semi-supervised node classification (AI-review [39], MentalNet-style).
- Can lift accuracy when labeled data is scarce, but breaks pure inductive deployment and risks label leakage across the similarity graph. Treat as an experiment, not the main model. If used, build the population graph **within the training fold only**.

### 6.7 Acoustic-event tokenization edges — optional novelty
Tokenize acoustic space into "acoustic words" (SDR-survey [140]) and add edges between segments sharing dominant tokens; or add a node-feature stream of acoustic-word histograms. Cheap, distinctive, worth one ablation.

**Recommended final architecture (v1 "hero" model):** baseline RGAT encoder **+ 6.1 hierarchical pooling + 6.2 contrastive pretraining + 6.3 anti-oversmoothing + 6.4 adversarial gender head**. That is a coherent, defensible research contribution that runs on the target laptop, and every piece traces to a named gap in the surveys.

---

## 7. Training strategy under the compute budget

- **PyTorch + PyTorch Geometric**, CPU device for the GNN. PyG on CPU is fine at this scale.
- Intel Arc: use `intel-extension-for-pytorch` (`ipex`) with `xpu` device **only** for the one-time wav2vec2 extraction pass if it helps; otherwise CPU. Do not rely on XPU for PyG (support is experimental).
- Batch = whole graphs, `torch_geometric.loader.DataLoader`, batch size 8–16 subjects. Peak RAM well under 16 GB.
- Mixed precision not needed (CPU). Gradient accumulation not needed.
- Full 5-fold CV of the hero model: target < 1 hour wall-clock. If longer, shrink hidden dim to 96 and Set B PCA to 64.
- Determinism: fix seeds, log everything (Hydra or a plain `config.yaml` + `wandb`/CSV). Save per-fold checkpoints.
- Augmentation is done **on cached features** (feature masking, mixup, Gaussian noise) — cheap. Waveform-level augmentation only during the offline extraction of extra minority-class views.

---

## 8. Evaluation protocol

- **Primary metric:** UAR (unweighted average recall / balanced accuracy) and macro-F1. Also report accuracy, precision, recall, specificity, AUC-ROC (metric definitions in AI-review §6).
- **Speaker-independent 5-fold CV**, stratified on label × gender. Report mean ± std.
- **Statistical testing:** McNemar or paired permutation test vs. the BiLSTM baseline; bootstrap 95 % CI on UAR (SDR-survey §V explicitly asks for this — most papers skip it).
- **Fairness slice:** metrics separately for male / female; report the gap.
- **Cross-corpus generalization (Phase 4):** train on DAIC-WOZ, test on E-DAIC (and optionally DEPAC) with zero fine-tuning — different collection protocol / elicitation, same language. Expect a drop; the story is how much 6.2 + 6.4 reduce it.
- **Ablation table:** baseline → +each component → hero model, on the same folds.
- **Feature-group ablation:** A vs. A+C (from P1, on the SVM/RF baseline and later the GNN), plus +B once P4 adds it. Also report each guide group alone (prosodic only, voice-quality only, …) on the SVM/RF baseline. This directly answers whether combining groups helps.
- **Interpretability artifacts:** per-subject segment-importance timeline (from 6.1 pooling), aggregate which acoustic features / regions matter, sanity-check against known markers (reduced F0 variability, slower rate, longer pauses and response latency, lower CPP).

---

## 9. Repository structure

Actual current layout (deviates slightly from the original sketch below — DAIC_WOZ and papers were moved to top level rather than nested under `data/`):

```
speech-detection/
├── DAIC_WOZ/                       # raw dataset — gitignore this directory (16 GB)
│   ├── audio/  covarep/  formant/  transcripts/  splits/
├── documents/                      # supporting papers (AVEC2017 dataset doc, SDR survey, COVAREP, PHQ8, etc.)
├── data/
│   └── cache/DAIC_WOZ/<subject>.pt # created in P0, gitignore this too
```

Original planned layout for reference (functionally equivalent — `data/raw/DAIC_WOZ` ≡ `DAIC_WOZ/`, adjust config paths accordingly):

```
speech-detection/
├── IMPLEMENTATION_PLAN.md          # this file
├── configs/
│   ├── data_daic.yaml
│   ├── data_edaic.yaml
│   ├── data_depac.yaml
│   └── model_hero.yaml
├── data/
│   ├── raw/                        # gitignored
│   └── cache/<dataset>/<subject>.pt
├── src/
│   ├── data/
│   │   ├── download.py             # dataset access helpers / instructions
│   │   ├── preprocess.py           # resample, VAD, segment
│   │   ├── features_egemaps.py     # Set A: eGeMAPS core
│   │   ├── features_prosody.py     # Set A extras: CPP, rhythm, pauses, words/sec, response latency
│   │   ├── features_linguistic.py  # Set C: participant-text features
│   │   ├── features_ssl.py         # Set B (wav2vec2/HuBERT, frozen)
│   │   └── build_graph.py          # edges: temporal / knn / recurrence
│   ├── models/
│   │   ├── encoder_rgat.py
│   │   ├── pooling.py              # SAGPool / ASAP
│   │   ├── pretrain_graphcl.py     # 6.2
│   │   ├── adversary.py            # 6.4 gradient reversal + gender head
│   │   ├── ib_subgraph.py          # 6.5
│   │   └── classifier.py           # full model assembly
│   ├── train/
│   │   ├── cv.py                   # speaker-independent folds
│   │   ├── loop.py
│   │   └── metrics.py
│   ├── eval/
│   │   ├── ablation.py
│   │   ├── cross_corpus.py
│   │   └── interpret.py            # segment-importance timelines
│   └── baselines/
│       ├── svm_rf.py
│       ├── bilstm.py
│       └── wav2vec_probe.py
├── notebooks/                      # exploration only
├── results/                        # metrics csv, figures, checkpoints
├── requirements.txt
└── README.md
```

Key deps: `torch`, `torch-geometric`, `opensmile`, `praat-parselmouth`, `transformers`, `torchaudio`, `silero-vad`, `scikit-learn`, `librosa`, `numpy<2` (compat), `pandas`, `matplotlib`, `vaderSentiment` or the NRC Emotion Lexicon file (Set C). Optional: `intel-extension-for-pytorch`.

---

## 10. Milestones

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0 — Setup** (wk 1) | License + DAIC-WOZ downloaded, `preprocess.py` (CSV path first), `build_graph.py` stub, cache built | Cached `.pt` for every DAIC subject from the shipped COVAREP + FORMANT CSVs (eGeMAPS appended once extracted); EDA notebook (label/gender/duration distributions) |
| **P1 — Baselines** (wk 2) | SVM/RF, BiLSTM/GRU, wav2vec2 linear probe, speaker-independent CV harness | Numbers in the ballpark of published DAIC-WOZ speech baselines (AVEC 2016); CV harness trusted |
| **P2 — GNN baseline** (wk 3) | `build_graph.py` + RGAT encoder + flat readout | GNN ≥ BiLSTM at equal features on speaker-independent UAR; **one-time ONNX/TorchScript export smoke-test of the RGAT block passes** (catch export-incompatible ops now, while the model is still simple — see §15) |
| **P3 — Architecture increments** (wk 4–6) | 6.1, 6.2, 6.3, 6.4 implemented + ablation table | Hero model beats P2 with statistical significance; fairness gap reported |
| **P4 — Generalization + interpretability** (wk 7–8) | raw-audio wav2vec2 stream, E-DAIC/DEPAC cross-corpus results, interpret.py figures | Cross-corpus numbers + segment-importance figures; results/ populated |
| **P5 — Optional depth** (wk 9+) | 6.5 IB subgraph, 6.6 population graph, 6.7 acoustic-word edges | Each kept only if it beats the hero model |
| **P6 — Write-up** | Paper / report draft, reproducibility package (code + configs + Docker) | SDR-survey §V reproducibility checklist satisfied — **this is "software project complete"; §14/§15 below start only after this** |
| **P7 — Optional video extension** (low priority) | Second visual-graph branch fused with the audio graph (§14) | Fused model beats audio-only hero model on the same speaker-independent folds, or is reported as a null result. Independent of P8 — may be skipped, deferred, or attempted after P8 |
| **P8 — Edge deployment** (primary post-P6 direction) | Quantized/exported audio-only "edge profile" model running on-device on Raspberry Pi and/or Jetson Nano (§15) | Meets a defined latency/RAM budget on-device with an accuracy drop you've explicitly measured and accepted. Depends only on P6, not P7 |

---

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| DAIC-WOZ small (189 subj) → unstable CV, high variance | High | Contrastive pretraining (6.2); repeat CV with 5 seeds; report CIs; pool DAIC + E-DAIC for training in P4 |
| DAIC 20-min interviews too heavy to featurize on the laptop | Medium | Start from shipped feature CSVs (no signal processing); cap nodes/session; if still heavy, switch primary to DEPAC (short structured clips) |
| GNN doesn't beat BiLSTM | Medium | Fix graph construction first (edge types, k, learnable adjacency); if still flat, the contribution becomes the *interpretability + fairness* framing, not raw accuracy |
| wav2vec2 extraction too slow / OOM on laptop | Medium | Use `-base` not `-large`; process in 30 s chunks; fall back to Set A only (must be a first-class path) |
| Model learns speaker identity, not depression | High (surveys stress this) | Speaker-independent splits; adversarial head (6.4); disjoint train/test speakers verified in code |
| Label noise (self-report PHQ-8) | Inherent | Report against threshold sensitivity; don't over-tune; note as limitation |
| Intel Arc / IPEX instability | Medium | Treat XPU as optional; CPU path is the supported one |
| Cross-corpus collapse (DAIC → E-DAIC / DEPAC: different protocol, elicitation, mic) | Medium–High | Expected; frame as the research question; domain-adversarial head; recording-condition normalization |

---

## 12. Component → survey-gap traceability

| Component | Survey gap it addresses |
|---|---|
| Multi-group features: prosodic, voice-quality, spectral, temporal, linguistic (§3) | Guide's guidance (2026-10-01) to combine complementary feature groups rather than rely on one; the feature-group ablation (§8) tests it |
| Multi-relational learnable graph (§4) | "No consensus on graph construction" (AI-review neuroimaging synthesis) |
| Hierarchical pooling + timelines (6.1) | Interpretability for clinical trust (SDR-survey §V; AI-review §7.2) |
| Contrastive graph pretraining (6.2) | Small/imbalanced datasets, overfitting (SDR-survey MC1; AI-review §7.4) |
| Anti-oversmoothing (6.3) | Known GNN failure mode (AI-review [40]) |
| Adversarial gender/speaker head (6.4) | Bias toward subpopulations, speaker-independent reps (SDR-survey KI3/KI7; AI-review §7.4) |
| IB subgraph (6.5) | Inherently-interpretable-by-design models (AI-review §7.2) |
| Cross-corpus eval (§8) | Cross-corpus / cross-protocol generalization, the SDR-survey's central open problem (§V; AI-review §7.5) |
| Bootstrap / permutation tests (§8) | Missing statistical rigor (SDR-survey §V) |
| Reproducibility package (P6) | Data+code+container redistribution (SDR-survey §V) |

---

## 13. Stretch / future (not v1)

- Symptom-level heads (PHQ-8 items) once binary is solid — both surveys argue the sum-score obscures MDD heterogeneity.
- Cross-lingual transfer: pretrain the graph encoder on English (DAIC), adapt to a non-English corpus (EATD/MODMA) with a small labelled set — tests the "language-agnostic depression representation" idea (AI-review §7.5). Not v1 because it reopens the multilingual can of worms.

(Video extension and edge deployment were also stretch ideas in earlier drafts of this plan; they're now confirmed future work and formalized as §14 and §15 rather than left here.)

---

## 14. Phase 7 — Optional video extension

**Priority: low.** Confirmed as a "slight possibility, not the main path" — the primary post-P6 direction is §15 (edge deployment), which this phase does **not** gate. Read the rest of this section only if/when you decide to actually attempt it; it's fine to skip permanently, attempt it after §15 instead of before, or abandon it partway once the audio-only edge deployment work is underway. Everything in §15 is written to be reachable straight from P6 with zero dependency on this section.

**Goal:** add a visual modality without redesigning the audio pipeline or backbone built in P0–P6.

**Why this is comparatively cheap:** DAIC-WOZ already ships per-session visual features (CLNF facial landmarks/3D, head pose, gaze, action units) for the *same* subjects and the *same* timeline you already segment for audio — the download script currently skips these (`_CLNF_*` files, ~115 GB across the corpus) precisely because P0–P6 don't need them. Adding video later means re-running `download_daic.py` with the CLNF suffixes included (or per-subject on demand), not acquiring a new dataset.

**Design — two-branch graph fusion, not a redesign:**
1. Build a **second per-subject graph** using the same segment timeline (3 s / 1.5 s hop) already established for audio, but with visual-descriptor node features (AU intensities, head-pose/gaze summary stats per window) instead of acoustic ones. Reuse `build_graph.py`'s edge-construction logic (temporal + kNN) unchanged — it's already modality-agnostic since it only looks at node feature vectors and timestamps.
2. Fuse the two branches at the **graph-embedding level**: run each graph through its own encoder (the existing `encoder_rgat.py` reused for both, different weights), then combine the two pooled embeddings before the final MLP head — concatenation is the simplest baseline; cross-modal attention (one embedding attends over the other's node states before pooling) is the stronger version and mirrors the "heterogeneous multimodal GNN" idea already named in the old §13 (AI-review [9] MS²-GNN) and the two-level population-graph pattern in §6.6.
3. Alternative if you want tighter fusion: instead of two separate graphs, add a same-timestep **cross-modal co-occurrence edge type** (`E_av`) directly into the existing multi-relational graph from §4, connecting the audio node and visual node for the same 3 s window. This fits the existing `edge_type` mechanism in §4/§5 without adding a second encoder — try this first since it's the smallest change to existing code.
4. **Evaluate as an ablation, not a replacement:** report audio-only (P6 hero model) vs. audio+video on the same speaker-independent folds. A null result (video doesn't help) is a legitimate, reportable outcome — DAIC-WOZ's video is lower ecological validity than a real deployed camera feed, so don't over-invest if the audio-only model was already strong.
5. Relevant reading already in `documents/`: `scherer_stratou_etal_2014.pdf` and `stratou_scherer_etal_2014.pdf` both describe USC's own audiovisual behavior-descriptor pipeline for this exact corpus — read these before building the visual feature extractor, since they define what the shipped CLNF/AU columns mean.

**Exit criterion:** fused model's speaker-independent UAR compared against the audio-only hero model with the same statistical test used in §8, plus a fairness slice — video adds new sources of bias (lighting, camera angle, appearance) worth checking.

---

## 15. Phase 8 — Edge deployment (Raspberry Pi / Jetson Nano)

**Priority: primary post-P6 direction.** This is the confirmed next step after the software project is complete, independent of whether §14 (video) ever happens. Everything below assumes the **audio-only** hero model from P6 as the thing being deployed.

**Goal:** get the finished audio-only model running on-device, after the software project is otherwise complete. This phase changes *what ships*, not the model you validate in the paper — keep the P6 "full" model as the reported result, and treat everything below as a separate, explicitly-degraded **edge profile**.

Skipping video (§14) isn't just lower priority, it actively helps here: a visual stream would require a camera on the device, real-time face detection/CLNF-equivalent extraction, and roughly doubles the on-device compute and power budget — all working against the latency/RAM constraints this phase is optimizing for. If §14 is ever revisited, treat it as a separate "audiovisual edge profile" fork evaluated on its own merits, not a default upgrade to the deployed model.

**The real bottleneck is Set B, not the GNN.** The RGAT backbone is already tiny (~200–400 K params, §5) and runs on CPU in minutes for the whole dataset — it was never the constraint. The frozen wav2vec2-base/HuBERT-base stream (Set B, §3) is ~95 M parameters and is the part that doesn't fit a Raspberry Pi (no GPU, ARM CPU only) or comfortably fit a Jetson Nano (128 CUDA cores, ~0.5 TFLOPS) at anything near real-time.

**Two deployment profiles:**
| | Full (P0–P6 result) | Edge profile (P8) |
|---|---|---|
| Features | Set A + Set C + Set B (§3) | **Set A only, minus words/sec** (~98-dim). openSMILE is C++ and Parselmouth is lightweight, so both run cheaply on Pi and Nano. Words/sec and Set C need transcripts, which would mean on-device speech recognition, so they're dropped; syllable rate already covers speech rate. Response latency is kept only if the device itself plays the interview questions, so it knows when each one ends. |
| GNN backbone | Hero model from §6 | Same architecture, hidden dim shrunk if needed (§7 already anticipates 96-dim as a fallback) |
| Where it runs | Laptop / cloud, offline | On-device |
| Expected accuracy | Reference number for the paper | Lower — report the exact gap, don't hide it (this is itself a legitimate result: "SSL features cost you a GPU; here's what you lose without one") |

If Jetson Nano specifically (not Pi) turns out to have enough headroom, Set B can be kept but only via a quantized/TensorRT-optimized wav2vec2, or a distilled small SSL model — decide this empirically on-device, don't assume it up front.

**Practical steps:**
1. **Export path:** TorchScript trace or ONNX export of the trained GNN backbone. Do the smoke-test in P2 (already added to §10's milestone table) rather than waiting until P8 — PyG's message-passing ops have historically had partial/version-dependent ONNX support, and finding this out after P3–P6 have built more architecture on top is expensive to unwind.
2. **Fixed shapes:** the existing node cap (≈256 nodes/subject, §2) is already export-friendly — ONNX/TensorRT strongly prefer static shapes over PyG's default variable-sized batches. Keep this cap explicit in the edge profile rather than letting it vary.
3. **On-device graph construction:** kNN edge construction (§4) has to be recomputed per new recording at inference time (there's no cache on a deployed device). At ≤256 nodes × ~98 dims this is trivial even on a Pi's Cortex-A72 — brute-force kNN, no approximate-NN library needed.
4. **Quantization:** INT8 post-training quantization of the GNN backbone once it's export-verified; re-check UAR after quantization against the fp32 edge-profile number before shipping.
5. **Benchmark:** define and hit a concrete latency/RAM budget per target board (e.g., end-to-end inference under N seconds on a 3-minute recording, under M MB resident) — pick N/M once you know which Pi/Nano model you're targeting; these differ enough (Pi 4 vs Pi 5, Nano vs Nano 2GB/Orin Nano) that it's not worth guessing now.

**Exit criterion:** edge-profile model runs end-to-end on the target board within the defined latency/RAM budget, with the accuracy gap vs. the full model measured and reported.
