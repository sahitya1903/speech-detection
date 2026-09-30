# Project Tracker

Task breakdown of [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), by phase. See IMPLEMENTATION_PLAN.md §0.5 for this semester's compressed timeline, team split, and dual-track presentation strategy — phase headings below carry target dates from that plan. The "Study & Execution Plan" PDF (v2, updated 2026-10-04) is the authoritative day-to-day calendar.

**Mid-Review is Fri 9 Oct 2026 (rescheduled from 26–30 Sep); 4–8 Oct is a Mid-Review sprint, so P0–P3 dates were re-baselined on 4 Oct (Study & Execution Plan v2 §5, §8).** **Real engineering deadline: P0–P3(core)+P8 done by ~5 Nov 2026 (unchanged).** Official End-Review (~7–9 Nov, before End-Sem exams 10–26 Nov) shows only P0+P1+basic P2; P3-full/P8 are finished but held back for next semester's official reveal. Paper (P6) starts 27 Nov+. P4/P5/P7 are a conditional stretch tier, attempted in that order only if ahead of schedule.

**How this file updates:** run `python scripts/update_tracker.py` after making progress (ask Claude to run it — it's cheap and safe to run anytime). It does two things:
1. Recomputes the progress table below from checkbox counts.
2. Re-checks every box tagged `<!--auto:KEY-->` against the actual repo state (file exists, session counts match, etc.) and **overwrites** that box's checked/unchecked state to match reality — auto-tagged boxes always reflect the filesystem, they're not meant to be hand-edited.

Boxes with **no** `<!--auto:...-->` tag are judgment calls (e.g. "beats BiLSTM with significance") — the script never touches these. Check/uncheck them yourself as you go.

<!-- PROGRESS_TABLE_START -->
| Phase | Done | Total | % |
|---|---|---|---|
| P0 — Setup (target: 13 Oct 2026 — **You**; was 7 Oct) | 1 | 6 | 17% |
| P1 — Baselines (target: 18 Oct 2026 — **Partner** leads; was 14 Oct) | 0 | 8 | 0% |
| P2 — GNN baseline (target: 22 Oct 2026 — **You** leads; was 21 Oct; this is the version shown at the official End-Review) | 0 | 6 | 0% |
| P3 — Architecture increments (target: 31 Oct 2026 — **You** leads; core items 6.1/6.3/6.4 only this semester, 6.2 only if ahead of schedule; finished but held back from the official End-Review deck) | 0 | 7 | 0% |
| P4 — Generalization + interpretability (STRETCH TIER #1 — only if ahead of schedule, after 5 Nov or in the 27 Nov+ gap; highest-value stretch item) | 0 | 6 | 0% |
| P5 — Optional depth (STRETCH TIER #2 — attempt sub-items in order below, only if ahead of schedule; each kept only if it beats the hero model) | 0 | 3 | 0% |
| P6 — Write-up (target: starts 27 Nov 2026 or later, after End-Sem exams — this is the final paper, distinct from the shorter Mid/End-Review written deliverables tracked above) | 0 | 3 | 0% |
| P7 — Optional video extension (STRETCH TIER #3 — lowest priority, highest cost; realistically only reachable in the 27 Nov+ gap; check free disk space before downloading ~75-135GB of CLNF/HOG features) | 0 | 7 | 0% |
| P8 — Edge deployment (Raspberry Pi / Jetson Nano) (target: 1-5 Nov 2026 — **Both**, joint; this semester this happens BEFORE P6 write-up, not after — demoed privately to mentor, revealed to panel next semester) | 0 | 9 | 0% |
<!-- PROGRESS_TABLE_END -->

*Last auto-check: 2026-10-03 19:34 UTC*

---

## Institutional Milestones & Presentation Strategy

Calendar-driven, not implementation-driven — tracked separately from the P0-P8 phase checklist below since it doesn't map 1:1 to the technical work. See IMPLEMENTATION_PLAN.md §0.5 for full rationale.

- [ ] Mid-Review written deliverables drafted (Problem Statement, Motivation, Lit. Review, Research Gap, Objectives, Methodology) — originally 15 Sep; **status to be confirmed 4 Oct**, finish by 8 Oct — **[Both]**
- [ ] Preliminary implementation for Mid-Review (pandas + sklearn baseline, no PyTorch yet) — `notebooks/01_mid_review_features_expanded.ipynb`, target 7 Oct 2026 — **[You]**
  - [x] Step 1: participant-only transcripts, labels, per-session summary (189 sessions)
  - [ ] Step 2: timing features from transcripts (words/sec, response latency, pauses, fillers) — 4 Oct
  - [ ] Step 3: COVAREP per-session mean/std on participant frames (voiced-only for F0) — 5 Oct
  - [ ] Step 4: depressed vs. non-depressed plots + Mann–Whitney U, effect size, BH-FDR — 6 Oct
  - [ ] Step 5: LR/SVM/RF with subject-level 5-fold CV (scaler fitted inside folds), UAR/macro-F1/AUC — 7 Oct
- [ ] Subject-level 5-fold split file (label × gender) saved as CSV — 6 Oct — **[Partner]**
- [ ] Mid-Review slides frozen + notebook Restart & Run All clean + two timed rehearsals (each presents the other's section) — 8 Oct — **[Both]**
- [ ] Mid-Review presented — **Fri 9 Oct 2026** (rescheduled from 26–30 Sep)
- [ ] End-Review deck assembled incrementally as P0/P1/P2 land — P0+P1+basic-P2 shown complete; P3-full and P8 named only under "Future Scope" — target 6 Nov 2026 — **[Both]**
- [ ] P8 edge deployment demoed privately to mentor (not the panel) — target 5 Nov 2026 — **[Both]**
- [ ] End-Review presented — target 7–9 Nov 2026 (before End-Sem Lab exams start 10 Nov)
- [ ] End-Sem exams survived (Lab 10–12 Nov, Theory 13–26 Nov) — full project blackout, no work expected
- [ ] Stretch tier attempted in order (P4 → P5 → P7) if ahead of schedule — optional, ongoing
- [ ] Final research paper drafted — starts 27 Nov 2026 or later
- [ ] P3-full results + P8 edge deployment officially revealed to the judging panel — next semester's final review

## P0 — Setup (target: 13 Oct 2026 — **You**; was 7 Oct)

- [x] DAIC-WOZ downloaded: audio + transcripts + COVAREP + FORMANT + splits, 189/189 sessions <!--auto:daic_downloaded-->
- [ ] `src/data/preprocess.py` written (Ellie-turn stripping, transcript-timestamp alignment) <!--auto:preprocess_written-->
- [ ] `src/data/build_graph.py` stub created (interface only, no edge logic yet) <!--auto:build_graph_stub-->
- [ ] Cache built: `data/cache/DAIC_WOZ/<subject>.pt` for every subject <!--auto:cache_built-->
- [ ] Set A extras appended to cache from raw audio + transcripts: CPP, syllable rate, articulation rate, speech-to-pause ratio, pause stats, F0 slope, words/sec, response latency — `src/data/features_prosody.py` (see plan §3 table)
- [ ] EDA notebook: label balance, gender split, interview-length distribution <!--auto:eda_notebook-->

## P1 — Baselines (target: 18 Oct 2026 — **Partner** leads; was 14 Oct)

- [ ] Speaker-independent 5-fold CV harness (stratified label × gender) <!--auto:cv_harness-->
- [ ] SVM/RF baseline (segment-mean eGeMAPS) <!--auto:svm_rf_baseline-->
- [ ] Set C linguistic features (participant turns only, never Ellie's text) — `src/data/features_linguistic.py`
- [ ] Feature-group ablation on SVM/RF: each guide group alone, Set A vs. Set A+C
- [ ] BiLSTM/GRU baseline <!--auto:bilstm_baseline-->
- [ ] wav2vec2 linear-probe baseline <!--auto:wav2vec_probe_baseline-->
- [ ] Baseline numbers checked against published AVEC 2016 DAIC-WOZ speech baselines
- [ ] CV harness manually verified to not leak speakers across folds

## P2 — GNN baseline (target: 22 Oct 2026 — **You** leads; was 21 Oct; this is the version shown at the official End-Review)

- [ ] `build_graph.py` edges implemented: temporal `E_seq`, kNN `E_knn`, optional recurrence `E_rec` <!--auto:graph_edges_impl-->
- [ ] `src/models/encoder_rgat.py`: RGATConv blocks + residual/LayerNorm <!--auto:rgat_encoder-->
- [ ] `src/models/classifier.py`: flat readout (mean/max/attention pool) + MLP head <!--auto:classifier_assembly-->
- [ ] GNN trained through the same CV harness as P1
- [ ] GNN ≥ BiLSTM at equal features on speaker-independent UAR
- [ ] ONNX/TorchScript export smoke-test of the RGAT block passes <!--auto:onnx_export_test-->

## P3 — Architecture increments (target: 31 Oct 2026 — **You** leads; core items 6.1/6.3/6.4 only this semester, 6.2 only if ahead of schedule; finished but held back from the official End-Review deck)

- [ ] 6.1 Hierarchical pooling (SAGPool/ASAP) — `src/models/pooling.py` <!--auto:pooling_impl-->
- [ ] 6.2 Contrastive graph pretraining (GraphCL) — `src/models/pretrain_graphcl.py` — only if ahead of schedule <!--auto:contrastive_pretrain_impl-->
- [ ] 6.3 Anti-oversmoothing controls (GIN/APPNP, jumping-knowledge, DropEdge)
- [ ] 6.4 Gender/speaker adversarial head — `src/models/adversary.py` <!--auto:adversary_impl-->
- [ ] Ablation table: baseline → +each component → hero model — `results/ablation_table.csv` <!--auto:ablation_table-->
- [ ] Hero model beats P2 with statistical significance (McNemar / paired permutation)
- [ ] Fairness gap (male vs. female UAR) reported

## P4 — Generalization + interpretability (STRETCH TIER #1 — only if ahead of schedule, after 5 Nov or in the 27 Nov+ gap; highest-value stretch item)

- [ ] Set B (wav2vec2/HuBERT) raw-audio feature stream added — `src/data/features_ssl.py` <!--auto:features_ssl_impl-->
- [ ] E-DAIC downloaded (only now, not before P4) <!--auto:edaic_downloaded-->
- [ ] DEPAC downloaded (optional) <!--auto:depac_downloaded-->
- [ ] Cross-corpus eval: train DAIC-WOZ → test E-DAIC (and optionally DEPAC), zero fine-tuning — `src/eval/cross_corpus.py` <!--auto:cross_corpus_impl-->
- [ ] `src/eval/interpret.py`: segment-importance timelines from pooling scores <!--auto:interpret_impl-->
- [ ] Cross-corpus numbers + interpretability figures populated in `results/`

## P5 — Optional depth (STRETCH TIER #2 — attempt sub-items in order below, only if ahead of schedule; each kept only if it beats the hero model)

- [ ] 6.5 Information-bottleneck subgraph selection — `src/models/ib_subgraph.py` <!--auto:ib_subgraph_impl-->
- [ ] 6.6 Two-level population graph (built within-training-fold only if used)
- [ ] 6.7 Acoustic-event tokenization edges

## P6 — Write-up (target: starts 27 Nov 2026 or later, after End-Sem exams — this is the final paper, distinct from the shorter Mid/End-Review written deliverables tracked above)

- [ ] Paper / report draft
- [ ] Reproducibility package (code + configs + Docker)
- [ ] SDR-survey §V reproducibility checklist satisfied

## P7 — Optional video extension (STRETCH TIER #3 — lowest priority, highest cost; realistically only reachable in the 27 Nov+ gap; check free disk space before downloading ~75-135GB of CLNF/HOG features)

- [ ] DAIC-WOZ CLNF/AU/gaze/pose features downloaded <!--auto:clnf_downloaded-->
- [ ] Read `scherer_stratou_etal_2014.pdf` / `stratou_scherer_etal_2014.pdf` (visual descriptor definitions)
- [ ] Visual feature extractor per 3 s segment window
- [ ] Try cross-modal co-occurrence edge type `E_av` in the existing multi-relational graph first
- [ ] (if needed) two-branch graph fusion with cross-modal attention
- [ ] Fused model evaluated vs. audio-only hero model on the same folds
- [ ] Fairness slice checked for video-introduced bias (lighting/camera/appearance)

## P8 — Edge deployment (Raspberry Pi / Jetson Nano) (target: 1-5 Nov 2026 — **Both**, joint; this semester this happens BEFORE P6 write-up, not after — demoed privately to mentor, revealed to panel next semester)

- [ ] Edge profile defined: Set A only, minus words/sec (no wav2vec2/HuBERT, no Set C linguistic, no transcript dependency)
- [ ] GNN backbone exported (ONNX/TorchScript) <!--auto:edge_export-->
- [ ] Fixed node-cap shapes verified in exported model
- [ ] On-device kNN edge construction implemented (no cache available at inference)
- [ ] INT8 post-training quantization + accuracy re-check vs. fp32 edge profile
- [ ] Benchmarked on Raspberry Pi <!--auto:benchmark_rpi-->
- [ ] Benchmarked on Jetson Nano <!--auto:benchmark_jetson-->
- [ ] Latency/RAM budget defined per target board and met
- [ ] Accuracy gap vs. full (P6) model measured and reported
