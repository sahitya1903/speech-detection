# Project Tracker

Task breakdown of [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), by phase. See IMPLEMENTATION_PLAN.md §0.5 for the two-semester timeline, team split, and presentation strategy.

**Mid-Review was Mon 12 Oct 2026. End-Review is 27 Nov 2026.**

**Two-semester scope:** Semester 1 (this semester) delivers P0 + P1 + P2. Semester 2 delivers P3 → P4 → P6 → P8 (with P5/P7 as stretch). No private demos — P8 is presented to the judging panel at next semester's final review.

**Status as of 25 Nov 2026:** Mid-Review done (12 Oct). **Semester 1 End-Review engineering & slides are complete** (for **27 Nov 2026**): P0 189-subject cache built (`data/cache/DAIC_WOZ/`), P1 5-fold speaker-independent CV + classical/BiLSTM baselines complete, P2 multi-relational RGAT GNN (`E_seq` + `E_knn`) trained & TorchScript export verified (`results/p2_export_smoke_test.ok`), `notebooks/02_end_review_pipeline.ipynb` executed clean, and `End_Review_Presentation.pptx` (13 slides) generated. Deferred to Dec break / Semester 2: raw-audio eGeMAPS, Set C linguistic features, wav2vec2 probe, and P3–P8.

**How this file updates:** run `python scripts/update_tracker.py` after making progress (ask Claude to run it — it's cheap and safe to run anytime). It does two things:
1. Recomputes the progress table below from checkbox counts.
2. Re-checks every box tagged `<!--auto:KEY-->` against the actual repo state (file exists, session counts match, etc.) and **overwrites** that box's checked/unchecked state to match reality — auto-tagged boxes always reflect the filesystem, they're not meant to be hand-edited.

Boxes with **no** `<!--auto:...-->` tag are judgment calls (e.g. "beats BiLSTM with significance") — the script never touches these. Check/uncheck them yourself as you go.

<!-- PROGRESS_TABLE_START -->
| Phase | Done | Total | % |
|---|---|---|---|
| P0 — Setup (Semester 1 — End-Review 27 Nov 2026 — **You**) | 7 | 7 | 100% |
| P1 — Baselines (Semester 1 — End-Review 27 Nov 2026 — **Partner** leads) | 7 | 8 | 88% |
| P2 — GNN baseline (Semester 1 — End-Review 27 Nov 2026 — **You** leads) | 5 | 6 | 83% |
| P3 — Architecture increments (Semester 2 — 6.1 hierarchical pooling, 6.3 anti-oversmoothing, 6.4 adversarial gender head are core; 6.2 contrastive pretraining only if ahead of schedule) | 0 | 7 | 0% |
| P4 — Generalization + interpretability (**Semester 2**, Jan–Mar 2027 — E-DAIC cross-corpus zero-shot eval + interpretability timelines) | 0 | 6 | 0% |
| P5 — Optional depth (STRETCH TIER #2 — attempt sub-items in order below, only if ahead of schedule; each kept only if it beats the hero model) | 0 | 3 | 0% |
| P6 — Write-up (target: starts 27 Nov 2026 or later, after End-Sem exams — this is the final paper, distinct from the shorter Mid/End-Review written deliverables tracked above) | 0 | 3 | 0% |
| P7 — Optional video extension (STRETCH TIER #3 — lowest priority, highest cost; realistically only reachable in the 27 Nov+ gap; check free disk space before downloading ~75-135GB of CLNF/HOG features) | 0 | 7 | 0% |
| P8 — Edge deployment (Raspberry Pi / Jetson Nano) (Semester 2 — presented to the judging panel at the final review; joint work by both students after P3 hero model is frozen) | 0 | 9 | 0% |
<!-- PROGRESS_TABLE_END -->

*Last auto-check: 2026-11-24 21:29 UTC*

---

## Institutional Milestones & Presentation Strategy

Calendar-driven, not implementation-driven — tracked separately from the P0-P8 phase checklist below since it doesn't map 1:1 to the technical work. See IMPLEMENTATION_PLAN.md §0.5 for full rationale.

**Semester 1**
- [x] Mid-Review written deliverables drafted (Problem Statement, Motivation, Lit. Review, Research Gap, Objectives, Methodology) — covered in `Mid_Review_Presentation.pptx` — **[Both]**
- [x] Preliminary implementation for Mid-Review (pandas + sklearn baseline, no PyTorch yet) — `notebooks/01_mid_review_features_expanded.ipynb`; results in `results/mid_review/` — **[You]**
  - [x] Step 1: participant-only transcripts, labels, per-session summary (189 sessions)
  - [x] Step 2: timing features from transcripts (words/sec, response latency, pauses, fillers)
  - [x] Step 3: COVAREP per-session mean/std on participant frames (voiced-only for F0)
  - [x] Step 4: depressed vs. non-depressed plots + Mann–Whitney U, effect size, BH-FDR (raw and gender-normalised, plus train-only stability check)
  - [x] Step 5: 12 classifiers on the official dev split + repeated stratified 5-fold CV (scaler fitted inside folds), UAR/macro-F1/AUC, bootstrap CIs, feature-group comparison, pruning/RFE, latency, AVEC 2016 comparison
- [x] Mid-Review slides frozen + notebook Restart & Run All clean (57 cells, sequential, no errors) + rehearsals — **[Both]**
- [x] Mid-Review presented — **Mon 12 Oct 2026**
- [x] Subject-level 5-fold split file (label × gender) saved as CSV — **[Partner]** — saved to `data/cache/DAIC_WOZ/folds_5cv.csv`
- [x] End-Review deck assembled (P0+P1+basic-P2 shown complete; P3–P8 under "Semester 2 / Future Work") — `End_Review_Presentation.pptx` (20 slides) — **[Both]**
- [ ] End-Review presented — **27 Nov 2026**

## Semester 2 (starts Dec 2026 — Final Review: Apr 2027)
- [ ] P3 architecture increments complete (hierarchical pooling, anti-oversmoothing, adversarial gender head; contrastive pretraining if ahead of schedule)
- [ ] P4 cross-corpus generalization complete (E-DAIC zero-shot eval + interpretability timelines)
- [ ] P6 final paper drafted + reproducibility package (code + configs + Docker)
- [ ] P8 edge deployment complete (quantized GNN on Raspberry Pi / Jetson Nano)
- [ ] Stretch tier attempted in order (P5 → P7) only if P3+P4+P6+P8 are ahead of schedule
- [ ] Final review: P3 full architecture + ablation table + P4 cross-corpus + P8 edge demo + paper presented to judging panel

## P0 — Setup (Semester 1 — End-Review 27 Nov 2026 — **You**)

- [x] DAIC-WOZ downloaded: audio + transcripts + COVAREP + FORMANT + splits, 189/189 sessions <!--auto:daic_downloaded-->
- [x] Environment: `torch`, `torch-geometric`, `opensmile`, `praat-parselmouth`, `vaderSentiment` installed in `speech-dep`; versions pinned in `requirements.txt`
- [x] `src/data/preprocess.py` written (Ellie-turn stripping, transcript-timestamp alignment) <!--auto:preprocess_written-->
- [x] `src/data/build_graph.py` stub created (interface only, no edge logic yet) <!--auto:build_graph_stub-->
- [x] Cache built: `data/cache/DAIC_WOZ/<subject>.pt` for every subject <!--auto:cache_built-->
- [x] Set A extras appended to cache from raw audio + transcripts: CPP, syllable rate, articulation rate, speech-to-pause ratio, pause stats, F0 slope, words/sec, response latency — `src/data/features_prosody.py` (see plan §3 table)
- [x] EDA notebook: label balance, gender split, interview-length distribution <!--auto:eda_notebook-->

## P1 — Baselines (Semester 1 — End-Review 27 Nov 2026 — **Partner** leads)

- [x] Speaker-independent 5-fold CV harness (stratified label × gender) <!--auto:cv_harness-->
- [x] SVM/RF baseline (segment-mean eGeMAPS) <!--auto:svm_rf_baseline-->
- [x] Set C linguistic features (participant turns only, never Ellie's text) — `src/data/features_linguistic.py`
- [x] Feature-group ablation on SVM/RF: each guide group alone, Set A vs. Set A+C (`results/end_review/feature_ablation_cv.csv`)
- [x] BiLSTM/GRU baseline <!--auto:bilstm_baseline-->
- [ ] wav2vec2 linear-probe baseline <!--auto:wav2vec_probe_baseline-->
- [x] Baseline numbers checked against published AVEC 2016 DAIC-WOZ speech baselines (compared in `End_Review_Presentation.pptx` Slide 9 & `results/mid_review/comparison_with_avec2016.csv`)
- [x] CV harness manually verified to not leak speakers across folds (`assert_no_speaker_leakage` in `src/train/cv.py`)

## P2 — GNN baseline (Semester 1 — End-Review 27 Nov 2026 — **You** leads)

- [x] `build_graph.py` edges implemented: temporal `E_seq`, kNN `E_knn`, optional recurrence `E_rec` <!--auto:graph_edges_impl-->
- [x] `src/models/encoder_rgat.py`: RGATConv blocks + residual/LayerNorm <!--auto:rgat_encoder-->
- [x] `src/models/classifier.py`: flat readout (mean/max/attention pool) + MLP head <!--auto:classifier_assembly-->
- [x] GNN trained through the same CV harness as P1 (`results/end_review/rgat_cv.csv`)
- [ ] GNN ≥ BiLSTM at equal features on speaker-independent UAR
- [x] ONNX/TorchScript export smoke-test of the RGAT block passes <!--auto:onnx_export_test-->

## P3 — Architecture increments (Semester 2 — 6.1 hierarchical pooling, 6.3 anti-oversmoothing, 6.4 adversarial gender head are core; 6.2 contrastive pretraining only if ahead of schedule)

- [ ] 6.1 Hierarchical pooling (SAGPool/ASAP) — `src/models/pooling.py` <!--auto:pooling_impl-->
- [ ] 6.2 Contrastive graph pretraining (GraphCL) — `src/models/pretrain_graphcl.py` — only if ahead of schedule <!--auto:contrastive_pretrain_impl-->
- [ ] 6.3 Anti-oversmoothing controls (GIN/APPNP, jumping-knowledge, DropEdge)
- [ ] 6.4 Gender/speaker adversarial head — `src/models/adversary.py` <!--auto:adversary_impl-->
- [ ] Ablation table: baseline → +each component → hero model — `results/ablation_table.csv` <!--auto:ablation_table-->
- [ ] Hero model beats P2 with statistical significance (McNemar / paired permutation)
- [ ] Fairness gap (male vs. female UAR) reported

## P4 — Generalization + interpretability (**Semester 2**, Jan–Mar 2027 — E-DAIC cross-corpus zero-shot eval + interpretability timelines)

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

## P8 — Edge deployment (Raspberry Pi / Jetson Nano) (Semester 2 — presented to the judging panel at the final review; joint work by both students after P3 hero model is frozen)

- [ ] Edge profile defined: Set A only, minus words/sec (no wav2vec2/HuBERT, no Set C linguistic, no transcript dependency)
- [ ] GNN backbone exported (ONNX/TorchScript) <!--auto:edge_export-->
- [ ] Fixed node-cap shapes verified in exported model
- [ ] On-device kNN edge construction implemented (no cache available at inference)
- [ ] INT8 post-training quantization + accuracy re-check vs. fp32 edge profile
- [ ] Benchmarked on Raspberry Pi <!--auto:benchmark_rpi-->
- [ ] Benchmarked on Jetson Nano <!--auto:benchmark_jetson-->
- [ ] Latency/RAM budget defined per target board and met
- [ ] Accuracy gap vs. full (P6) model measured and reported
