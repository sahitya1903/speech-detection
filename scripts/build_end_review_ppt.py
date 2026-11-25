"""
Generate End_Review_Presentation.pptx (20 slides) for the 27 Nov 2026 End-Review.

Designed with balanced card heights, 13.5-15 pt body text (no half-empty boxes),
and dedicated full-width slides for all plots so every chart is crisp and legible.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES_END = ROOT / "results" / "end_review"
RES_MID = ROOT / "results" / "mid_review"
OUT_PPT = ROOT / "End_Review_Presentation.pptx"

NAVY = RGBColor(15, 35, 66)
TEAL = RGBColor(13, 110, 138)
EMERALD = RGBColor(25, 135, 84)
SLATE = RGBColor(71, 85, 105)
DARK = RGBColor(30, 41, 59)
LIGHT_BG = RGBColor(248, 250, 252)
CARD_BG = RGBColor(241, 245, 249)
BORDER = RGBColor(203, 213, 225)
WHITE = RGBColor(255, 255, 255)
AMBER = RGBColor(180, 83, 9)


def new_prs() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def add_bg(slide, color=LIGHT_BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_header(slide, title: str, subtitle: str = "", page_num: int | None = None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.26), Inches(11.9), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Calibri"
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = NAVY

    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.name = "Calibri"
        p2.font.size = Pt(13.5)
        p2.font.color.rgb = SLATE

    if page_num is not None:
        pb = slide.shapes.add_textbox(Inches(12.3), Inches(7.05), Inches(0.7), Inches(0.3))
        pf = pb.text_frame.paragraphs[0]
        pf.text = str(page_num)
        pf.alignment = PP_ALIGN.RIGHT
        pf.font.name = "Calibri"
        pf.font.size = Pt(11)
        pf.font.color.rgb = SLATE


def add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    body_lines: list[str],
    accent=TEAL,
    bg=WHITE,
    font_size: float = 14.0,
    title_size: float = 16.5,
):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = bg
    box.line.color.rgb = BORDER
    box.line.width = Pt(1)

    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(0.09), Inches(height))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.22), Inches(top + 0.14), Inches(width - 0.36), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = "Calibri"
    p.font.size = Pt(title_size)
    p.font.bold = True
    p.font.color.rgb = NAVY

    for line in body_lines:
        pl = tf.add_paragraph()
        pl.text = line
        pl.space_before = Pt(6)
        pl.line_spacing = 1.15
        pl.font.name = "Calibri"
        pl.font.size = Pt(font_size)
        pl.font.color.rgb = DARK


def add_kpi(slide, left: float, top: float, width: float, height: float, value: str, label: str, accent=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = WHITE
    box.line.color.rgb = BORDER
    box.line.width = Pt(1)

    tb = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.12), Inches(width - 0.32), Inches(height - 0.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.text = value
    p1.font.name = "Calibri"
    p1.font.size = Pt(27)
    p1.font.bold = True
    p1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.text = label
    p2.space_before = Pt(3)
    p2.font.name = "Calibri"
    p2.font.size = Pt(12.5)
    p2.font.color.rgb = SLATE


def add_flow_box(slide, left: float, top: float, width: float, height: float, title: str, sub: str, fill_color=TEAL):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.fill.background()

    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = title
    p.alignment = PP_ALIGN.CENTER
    p.font.name = "Calibri"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = sub
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(3)
    p2.font.name = "Calibri"
    p2.font.size = Pt(11.5)
    p2.font.color.rgb = RGBColor(226, 232, 240)


def add_table(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    headers: list[str],
    rows: list[list[str]],
    highlight_rows: set[int] | None = None,
    font_size: float = 12.5,
):
    tbl_shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(left), Inches(top), Inches(width), Inches(height))
    tbl = tbl_shape.table
    highlight_rows = highlight_rows or set()

    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = "Calibri"
        p.font.size = Pt(font_size)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER

    for r_idx, row in enumerate(rows):
        is_hi = r_idx in highlight_rows
        bg = RGBColor(232, 245, 238) if is_hi else (WHITE if r_idx % 2 == 0 else CARD_BG)
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = "Calibri"
            p.font.size = Pt(font_size)
            p.font.bold = is_hi
            p.font.color.rgb = DARK
            p.alignment = PP_ALIGN.LEFT if c_idx == 0 else PP_ALIGN.CENTER


def build_deck() -> None:
    prs = new_prs()
    blank = prs.slide_layouts[6]

    summary_df = pd.read_csv(RES_END / "summary_comparison.csv")
    fairness_df = pd.read_csv(RES_END / "fairness_gender.csv")
    ablation_df = pd.read_csv(RES_END / "feature_ablation_cv.csv")

    # ── Slide 1: Title ────────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s, NAVY)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.25), Inches(7.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    tb = s.shapes.add_textbox(Inches(0.9), Inches(1.4), Inches(11.5), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Speech-Based Depression Detection\nusing Graph Neural Networks"
    p.font.name = "Calibri"
    p.font.size = Pt(42)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "End-Review Presentation  ·  27 November 2026"
    p2.space_before = Pt(16)
    p2.font.name = "Calibri"
    p2.font.size = Pt(21)
    p2.font.color.rgb = RGBColor(125, 211, 252)

    p3 = tf.add_paragraph()
    p3.text = "Department of Electronics and Communication Engineering\nNational Institute of Technology Andhra Pradesh"
    p3.space_before = Pt(28)
    p3.font.name = "Calibri"
    p3.font.size = Pt(16)
    p3.font.color.rgb = RGBColor(203, 213, 225)

    p4 = tf.add_paragraph()
    p4.text = "Guide: Dr. V. P. Singh\nTeam: Sahitya Kushwaha (623195)  ·  Buddhana Pallavi (623106)  ·  Potti Pothan Abhishek (623156)"
    p4.space_before = Pt(22)
    p4.font.name = "Calibri"
    p4.font.size = Pt(15)
    p4.font.color.rgb = WHITE

    # ── Slide 2: Outline ──────────────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Outline", "End-Review Presentation Structure (20 Slides)", 2)
    outline_items = [
        ("01", "Introduction, Motivation & Objectives", "MDD burden, speech biomarkers, two-semester scope"),
        ("02", "Literature Survey & Research Gaps", "AVEC 2016, 2025 surveys, speaker leakage, temporal loss"),
        ("03", "Dataset, P0 Pipeline & Feature Sections", "DAIC-WOZ (189), 3 s windowing, Set A (eGeMAPS + prosody) & Set C"),
        ("04", "Methodology: P1 Baselines & P2 RGAT GNN", "5-fold speaker-independent CV, BiLSTM, 3-relation RGAT"),
        ("05", "Results, Ablation, Confusion Matrices & Future Scope", "CV comparisons, fairness slice, and Semester 2 roadmap"),
    ]
    for idx, (num, head, desc) in enumerate(outline_items):
        add_card(s, 0.7, 1.45 + idx * 1.1, 11.9, 0.94, f"{num}   {head}", [desc], accent=TEAL if idx < 4 else EMERALD, font_size=13.5)

    # ── Slide 3: Introduction & Motivation ────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Introduction & Motivation", "Why passive speech-based screening matters", 3)
    add_kpi(s, 0.6, 1.4, 2.8, 1.4, "322 M", "People globally living with\nMajor Depressive Disorder (WHO)")
    add_kpi(s, 3.6, 1.4, 2.8, 1.4, "4.4%", "Global prevalence; majority\nremain unscreened")
    add_kpi(s, 6.6, 1.4, 2.8, 1.4, "PHQ-8 ≥ 10", "Standard clinical threshold\nfor binary MDD screening")
    add_kpi(s, 9.6, 1.4, 3.1, 1.4, "1 Mic", "Only hardware required for\nnon-invasive deployment", accent=EMERALD)

    add_card(
        s, 0.6, 3.1, 5.9, 3.55,
        "Clinical & Acoustic Motivation",
        [
            "• Current diagnosis relies on subjective self-report questionnaires (PHQ-8/9) and scarce clinical interviews.",
            "• Psychomotor retardation in MDD measurably slows articulation, lengthens pauses & response latency, flattens pitch (F0), and alters glottal voice quality (NAQ, H1–H2, CPPS).",
            "• Speech offers low-cost, repeatable, and privacy-preserving screening.",
        ],
        font_size=14.0,
    )
    add_card(
        s, 6.8, 3.1, 5.9, 3.55,
        "Key Mid-Review Recap (12 Oct 2026)",
        [
            "• Built leak-free session-level baseline on 25 timing + COVAREP features.",
            "• Response latency was the strongest marker (1.02 s vs 0.83 s, p = 0.016 FDR-corrected).",
            "• Best Mid-Review CV UAR: 0.573 (Gaussian NB), beating AVEC 2016 audio F1.",
            "• Limitation: session averages erase temporal dynamics across the interview.",
        ],
        accent=AMBER,
        font_size=14.0,
    )

    # ── Slide 4: Problem Statement & Objectives ───────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Problem Statement & Objectives", "Two-semester capstone plan: Semester 1 (P0–P2) and Semester 2 (P3–P8)", 4)
    add_card(
        s, 0.6, 1.4, 12.1, 1.5,
        "Problem Statement",
        [
            "Given a participant's clinical interview speech (virtual agent turns stripped), classify whether the speaker is depressed (PHQ-8 ≥ 10) or not, evaluated strictly speaker-independently so zero participants overlap between training and test folds.",
        ],
        accent=NAVY,
        font_size=14.5,
    )
    objs = [
        ("O1 — Leak-Free Pipeline & Baselines (P0 + P1)  ✅ Complete", "Build a cached 3.0 s segment pipeline (COVAREP + Set A + Set C) and benchmark classical ML + BiLSTM under 5-fold speaker-independent CV."),
        ("O2 — Multi-Relational Speech Graph & GNN (P2)  ✅ Complete", "Construct per-subject speech graphs with temporal (E_seq), similarity (E_knn), and recurrence (E_rec) edges classified by a 2-layer RGAT encoder."),
        ("O3 — Architecture, Cross-Corpus & Edge (Sem 2 — Apr 2027)", "Add hierarchical pooling, anti-oversmoothing, and gender adversarial head (P3), E-DAIC cross-corpus eval (P4), final paper (P6), and Pi/Jetson deployment (P8)."),
    ]
    for idx, (t, d) in enumerate(objs):
        add_card(s, 0.6, 3.15 + idx * 1.3, 12.1, 1.12, t, [d], accent=EMERALD if idx < 2 else TEAL, font_size=13.5)

    # ── Slide 5: Literature Survey & Research Gaps ────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Literature Survey & Research Gaps", "Grounded in AVEC 2016 and two 2025 surveys (IEEE TAFFC & arXiv 2508.12022)", 5)
    lit_rows = [
        ["Valstar et al. (AVEC 2016) [1]", "COVAREP + linear SVM (frame vote)", "Dev F1: 0.462 (dep) / 0.682 (ctrl)", "Official DAIC-WOZ audio benchmark"],
        ["Leal et al. (IEEE TAFFC 2025) [2]", "Comprehensive SDR survey", "eGeMAPS + SSL reference", "Flags speaker leakage & cross-corpus gap"],
        ["Aleagha et al. (arXiv 2025) [3]", "AI / GNN depression review", "Graph + hierarchical pooling", "Flags small-N overfitting & interpretability"],
        ["Gratch et al. / Degottex et al. [4,5]", "DAIC-WOZ corpus & COVAREP", "189 interviews, 100 Hz features", "Glottal & formant acoustic foundation"],
    ]
    add_table(s, 0.6, 1.4, 12.1, 2.75, ["Source", "Approach / Scope", "Key Benchmark", "Relevance to Our Design"], lit_rows, font_size=12.5)

    add_card(s, 0.6, 4.45, 3.85, 2.35, "Gap 1: Optimistic Splits", ["Random/segment splits leak speaker identity (inflating accuracy >90%). We enforce strict subject-level 5-fold CV."], accent=AMBER, font_size=13.5)
    add_card(s, 4.72, 4.45, 3.85, 2.35, "Gap 2: Lost Temporal Structure", ["Session averages erase pause/rhythm dynamics, and surveys note no consensus on speech graph construction."], accent=TEAL, font_size=13.5)
    add_card(s, 8.85, 4.45, 3.85, 2.35, "Gap 3: Bias & Metric Traps", ["~30% class imbalance makes accuracy misleading; pitch correlates with sex (36% vs 25% depressed)."], accent=NAVY, font_size=13.5)

    # ── Slide 6: Dataset & Mid-Review Statistical Findings ────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Dataset (DAIC-WOZ) & Acoustic Marker Validation", "189 clinical interviews (56 depressed / 133 control); gender is a major confounder", 6)
    if (RES_MID / "top_features_boxplots.png").exists():
        s.shapes.add_picture(str(RES_MID / "top_features_boxplots.png"), Inches(0.55), Inches(1.4), width=Inches(6.2))
    if (RES_MID / "pitch_gender_confound.png").exists():
        s.shapes.add_picture(str(RES_MID / "pitch_gender_confound.png"), Inches(6.9), Inches(1.4), width=Inches(5.85))
    add_card(
        s, 0.55, 5.55, 12.2, 1.25,
        "Why Gender Stratification & Disentanglement Matter",
        [
            "• Women have higher depression prevalence in DAIC-WOZ (35.6% vs 24.5%) and higher baseline F0 (178 Hz vs 101 Hz).",
            "• Therefore, CV folds are stratified on (label × gender), and we audit male vs. female UAR separately.",
        ],
        accent=AMBER,
        font_size=13.5,
    )

    # ── Slide 7: P0 Leak-Free Pipeline ────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "P0 — Leak-Free Data Pipeline & Tensor Cache", "src/data/preprocess.py  +  features_prosody.py  +  features_linguistic.py", 7)

    steps = [
        ("1. Raw DAIC-WOZ", "189 Audio (.wav), COVAREP,\nFORMANT, Transcripts", NAVY),
        ("2. Strip Ellie", "Keep Participant turns only\n(zero label leakage)", TEAL),
        ("3. 3 s Windowing", "W = 3.0 s, Hop = 1.5 s\nCap ≤ 256 nodes (avg 221.4)", TEAL),
        ("4. Multi-Stream", "COVAREP (158) + Set A (99)\n+ Set C Linguistic (8)", EMERALD),
        ("5. Disk Cache", "data/cache/DAIC_WOZ/\n<subject_id>.pt (189 files)", NAVY),
    ]
    for i, (t, sub, col) in enumerate(steps):
        add_flow_box(s, 0.55 + i * 2.5, 1.45, 2.25, 1.35, t, sub, fill_color=col)

    add_kpi(s, 0.55, 3.15, 3.85, 1.4, "189 / 189", "Cached .pt subject files\n(107 train + 35 dev + 47 test)")
    add_kpi(s, 4.72, 3.15, 3.85, 1.4, "40 – 256", "Nodes per subject graph\n(avg 221.4 × 3.0 s windows)")
    add_kpi(s, 8.9, 3.15, 3.85, 1.4, "265 dims", "Total cached features/node\n(158 + 88 + 11 + 8)")

    add_card(
        s, 0.55, 4.85, 12.2, 1.95,
        "Engineering Rationale for Offline Caching",
        [
            "• Heavy acoustic/prosodic extraction (openSMILE + Praat CPPS) runs once offline and caches tensors to disk.",
            "• Training loops load precomputed .pt files in < 1 second, enabling fast 5-fold CV on a 16 GB CPU-only laptop.",
            "• Modular streams (x, x_egemaps, x_prosody, x_set_c) allow instant ablation and edge-profile selection.",
        ],
        font_size=13.5,
    )

    # ── Slide 8: Feature Sections ─────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Feature Sections: Complementary Acoustic & Linguistic Groups", "Combining prosodic, voice-quality, spectral, temporal, and linguistic cues", 8)
    feat_rows = [
        ["COVAREP + FORMANT", "Shipped 100 Hz CSVs (mean + std, voiced-only F0 via VUV > 0.5)", "158", "Segment", "preprocess.py"],
        ["Set A — eGeMAPS v02", "openSMILE functionals: F0 semitones, loudness, jitter, shimmer, HNR, MFCC 1–4", "88", "Segment", "features_prosody.py"],
        ["Set A — Prosody & CPP", "Praat syllable/articulation rate, pauses, F0 slope, CPPS, words/sec, latency", "11", "Seg/Turn", "features_prosody.py"],
        ["Set C — Linguistic", "Participant-only 1sg/1pl pronouns, VADER emotion, absolutist, fillers, tags", "8", "Turn→Seg", "features_linguistic.py"],
        ["Graph Structure", "Temporal (E_seq), cosine k-NN (E_knn), and recurrence (E_rec) edges", "3 rels", "Graph", "build_graph.py"],
    ]
    add_table(s, 0.6, 1.4, 12.1, 3.3, ["Feature Group", "Extracted Descriptors", "Dims", "Scale", "Module"], feat_rows, highlight_rows={1, 2, 3}, font_size=12.5)

    add_card(
        s, 0.6, 5.0, 12.1, 1.85,
        "Strict Anti-Leakage Guarantees",
        [
            "• Set C uses participant transcripts only — Ellie's follow-up prompts adapt to the participant and leak the label.",
            "• StandardScaler is fitted strictly on training-fold segments inside each CV fold and applied to validation segments.",
            "• Turn-level measures (speech rate, response latency, Set C) are mapped cleanly via cached turn_ids onto 3 s segments.",
        ],
        accent=EMERALD,
        font_size=13.5,
    )

    # ── Slide 9: Methodology — P1 Baselines ───────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Methodology: P1 Speaker-Independent CV & Baselines", "src/train/cv.py  +  src/baselines/svm_rf.py  +  src/baselines/bilstm.py", 9)
    add_card(
        s, 0.6, 1.4, 5.9, 2.65,
        "Unified 5-Fold CV Harness (folds_5cv.csv)",
        [
            "• Stratified 5-fold CV on (label × gender) across all 189 subjects.",
            "• assert_no_speaker_leakage() verifies disjoint train/val IDs.",
            "• Metrics: UAR (balanced accuracy), Macro-F1, Depressed F1, ROC-AUC, and 2,000-sample bootstrap 95% CIs.",
        ],
        font_size=13.5,
    )
    add_card(
        s, 6.8, 1.4, 5.9, 2.65,
        "Classical Baselines (Segment-Mean Vectors)",
        [
            "• SVM (RBF, class_weight='balanced'), Random Forest (300 trees, depth 6), Logistic Regression, and Gaussian Naive Bayes.",
            "• Evaluated across 5 feature sets (covarep, egemaps, set_a, set_a_c, all) for systematic feature ablation.",
        ],
        font_size=13.5,
    )
    add_card(
        s, 0.6, 4.3, 12.1, 2.55,
        "Sequential Baseline: 2-Layer Bidirectional LSTM",
        [
            "• Input: Variable-length sequence of 3.0 s segments [n_nodes, 107] (Set A + Set C).",
            "• Encoder: Linear(107 → 64) + ReLU + Dropout(0.3) → 2-layer BiLSTM (hidden=64, packed sequences).",
            "• Readout & Optimization: Masked mean + max pooling (256-dim) → MLP(256 → 64 → 1); weighted BCE (pos_weight = n_neg/n_pos), AdamW (lr=1e-3), CosineAnnealingLR (30 epochs), best val UAR checkpointing.",
        ],
        accent=EMERALD,
        font_size=13.5,
    )

    # ── Slide 10: Methodology — P2 Graph Construction ─────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Methodology: P2 Multi-Relational Speech Graph Construction", "src/data/build_graph.py — modeling both local timing and global prosodic similarity", 10)
    add_card(
        s, 0.6, 1.45, 3.85, 3.25,
        "E_seq (Edge Type 0)\nTemporal Sequence",
        [
            "• Bidirectional edges connecting consecutive segments i ↔ i+1.",
            "• Captures speech rate transitions, pause rhythm, and psychomotor slowing over time.",
        ],
        accent=NAVY,
        font_size=14.0,
    )
    add_card(
        s, 4.72, 1.45, 3.85, 3.25,
        "E_knn (Edge Type 1)\nAcoustic Similarity",
        [
            "• Undirected cosine k-NN (k = 5) in standardized 107-dim feature space.",
            "• Links prosodically similar segments (e.g., flat-affect moments) regardless of time.",
        ],
        accent=TEAL,
        font_size=14.0,
    )
    add_card(
        s, 8.85, 1.45, 3.85, 3.25,
        "E_rec (Edge Type 2)\nRecurrence Structure",
        [
            "• Non-adjacent edges (lag ≥ 3) exceeding the 95th-percentile cosine similarity.",
            "• Captures repetitive prosodic patterns across the interview.",
        ],
        accent=EMERALD,
        font_size=14.0,
    )
    add_card(
        s, 0.6, 5.0, 12.1, 1.8,
        "Leak-Free Graph Building",
        [
            "• Standardized node features (using training-fold StandardScaler only) are used to compute E_knn and E_rec inside each fold.",
            "• Pure-PyTorch GraphData & collate_graphs() batch variable-length graphs without external C-extension dependencies.",
        ],
        accent=SLATE,
        font_size=13.5,
    )

    # ── Slide 11: Methodology — P2 RGAT Architecture ──────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Methodology: P2 RGAT Architecture & Export Readiness", "src/models/encoder_rgat.py  +  src/models/classifier.py", 11)

    arch_blocks = [
        ("Input X", "[N, 107]\nSet A + Set C", NAVY),
        ("Projection", "Linear(107→128)\nReLU + Drop(0.3)", TEAL),
        ("RGAT × 2", "4 heads, 3 rels\nRes + LayerNorm", EMERALD),
        ("Flat Readout", "mean | max | attn\n[384-dim]", TEAL),
        ("MLP Head", "384 → 64 → 1\nP(depressed)", NAVY),
    ]
    for i, (t, sub, col) in enumerate(arch_blocks):
        add_flow_box(s, 0.55 + i * 2.5, 1.45, 2.25, 1.35, t, sub, fill_color=col)

    add_card(
        s, 0.55, 3.15, 5.95, 3.65,
        "Relational Graph Attention Mechanism",
        [
            "• Keys and values are conditioned on learned relation embeddings for (E_seq, E_knn, E_rec).",
            "• Per-destination-node edge softmax computes relation-aware attention weights across 4 heads.",
            "• Residual connections + LayerNorm stabilize training on small graphs (40–256 nodes).",
        ],
        font_size=14.0,
    )
    add_card(
        s, 6.8, 3.15, 5.95, 3.65,
        "Training & Edge-Export Smoke-Test",
        [
            "• Trained on the exact same 5 folds (50 epochs, AdamW lr=7e-4, weighted BCE, best val UAR).",
            "• TorchScript trace verified on fixed 256-node shape (results/p2_export_smoke_test.ok, max diff < 1e-5).",
            "• Lightweight footprint (~250K params) — ready for Semester 2 INT8 quantization & Pi/Jetson deployment.",
        ],
        accent=EMERALD,
        font_size=14.0,
    )

    # ── Slide 12: Results Table ───────────────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Results: 5-Fold Speaker-Independent CV (Set A+C, N = 189)", "Sequence & graph models outperform all static baselines with 95% CIs above chance", 12)
    rows = []
    for r in summary_df.itertuples(index=False):
        rows.append([
            r.model,
            f"{r.uar_mean:.3f} ± {r.uar_std:.3f}",
            f"{r.uar_oof:.3f} [{r.uar_ci95_low:.3f}, {r.uar_ci95_high:.3f}]",
            f"{r.macro_f1_mean:.3f}",
            f"{r.f1_dep_mean:.3f}",
            f"{r.roc_auc_mean:.3f}",
        ])
    add_table(s, 0.6, 1.45, 12.1, 3.8, ["Model", "UAR (Mean ± Std)", "OOF UAR [95% CI]", "Macro-F1", "Depressed F1", "ROC-AUC"], rows, highlight_rows={4, 5}, font_size=13.0)

    add_card(
        s, 0.6, 5.5, 12.1, 1.35,
        "Key Findings",
        [
            "• BiLSTM achieves 0.642 ± 0.047 UAR (Macro-F1 0.620, Dep F1 0.498, AUC 0.635), with 95% CI [0.565, 0.712] well above 0.50 chance.",
            "• RGAT GNN achieves 0.589 ± 0.064 UAR (Macro-F1 0.586), also strictly above chance [0.514, 0.662], beating all 4 classical models.",
        ],
        accent=EMERALD,
        font_size=13.5,
    )

    # ── Slide 13: Results Plot (Full-Size UAR Comparison) ─────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Results: UAR Comparison with 95% Bootstrap Confidence Intervals", "Out-of-fold UAR across all 189 subjects (2,000-resample bootstrap)", 13)
    s.shapes.add_picture(str(RES_END / "uar_comparison.png"), Inches(0.6), Inches(1.4), width=Inches(8.2))
    add_card(
        s, 9.0, 1.45, 3.75, 5.3,
        "Chart Takeaways",
        [
            "• Chance floor = 0.500 (red dashed line).",
            "• SVM (RBF) collapses to majority class (0.500 UAR) on high-dim means.",
            "• LogReg (0.568) & Gaussian NB (0.560) cross 0.50 but CIs include chance.",
            "• Both BiLSTM (0.639 OOF) & RGAT GNN (0.589 OOF) have 95% CIs strictly above 0.50.",
        ],
        accent=TEAL,
        font_size=13.5,
    )

    # ── Slide 14: Comparison with Methods & Feature Ablation ──────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Comparison Across Feature Sets & Published Benchmarks", "Combining acoustic (Set A) + linguistic (Set C) + COVAREP consistently improves UAR", 14)

    best_per_fs = ablation_df.sort_values("uar_mean", ascending=False).groupby("feature_set", sort=False).first().reset_index()
    fs_order = {"covarep": 0, "egemaps": 1, "set_a": 2, "set_a_c": 3, "all": 4}
    best_per_fs["ord"] = best_per_fs["feature_set"].map(fs_order)
    best_per_fs = best_per_fs.sort_values("ord")

    prog_rows = [["AVEC 2016 Audio SVM [1]", "Official Dev (N=35)", "—", "—", "0.462"]]
    for r in best_per_fs.itertuples(index=False):
        prog_rows.append([f"{r.feature_set} ({r.model})", "5-fold CV (N=189)", f"{r.uar_mean:.3f} ± {r.uar_std:.3f}", f"{r.macro_f1_mean:.3f}", f"{r.f1_dep_mean:.3f}"])
    prog_rows.append(["Set A+C (BiLSTM)", "5-fold CV (N=189)", "0.642 ± 0.047", "0.620", "0.498"])
    prog_rows.append(["Set A+C (RGAT GNN)", "5-fold CV (N=189)", "0.589 ± 0.064", "0.586", "0.418"])

    add_table(s, 0.6, 1.4, 12.1, 4.1, ["Feature Set / Method", "Evaluation", "UAR (Mean ± Std)", "Macro-F1", "Depressed F1"], prog_rows, highlight_rows={5, 6, 7}, font_size=12.5)
    add_card(
        s, 0.6, 5.7, 12.1, 1.2,
        "Ablation Summary",
        [
            "• Best classical model climbs steadily: 0.550 (COVAREP) → 0.564 (eGeMAPS) → 0.570 (Set A+C) → 0.592 (All features combined).",
            "• BiLSTM on Set A+C reaches 0.498 Depressed F1 (0.642 UAR), outperforming the AVEC 2016 audio SVM baseline (0.462 F1).",
        ],
        accent=EMERALD,
        font_size=13.0,
    )

    # ── Slide 15: Feature Importance (Full-Size Plot) ─────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Feature Importance & Marker Consistency", "Consensus top features across folds align with clinical depression literature", 15)
    if (RES_MID / "consensus_feature_importance.png").exists():
        s.shapes.add_picture(str(RES_MID / "consensus_feature_importance.png"), Inches(0.6), Inches(1.35), width=Inches(8.1))
    add_card(
        s, 8.95, 1.45, 3.8, 5.3,
        "Clinical Alignment",
        [
            "• Timing: slower speech, longer pauses, and delayed response latency reflect psychomotor retardation.",
            "• Voice Quality: H1–H2, Rd, and QOQ capture breathiness and glottal tension.",
            "• Spectral & Prosody: lower F0 variability and reduced CPPS reflect flat affect.",
            "• Linguistic (Set C): higher 1st-person singular pronouns & negative emotion rates.",
        ],
        accent=TEAL,
        font_size=13.5,
    )

    # ── Slide 16: Training Dynamics (Full-Size Plot) ──────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Training Dynamics: BiLSTM vs. RGAT GNN (Fold 0)", "Convergence behavior under weighted BCE and cosine learning-rate decay", 16)
    s.shapes.add_picture(str(RES_END / "training_curves_fold0.png"), Inches(0.6), Inches(1.4), width=Inches(12.1))
    add_card(
        s, 0.6, 5.65, 12.1, 1.25,
        "Why Flat-Readout RGAT Trails BiLSTM in Semester 1",
        [
            "• BiLSTM learns smooth sequential gating over 221 segments (0.642 UAR), whereas flat mean+max readout in P2 RGAT aggregates all 221 nodes equally.",
            "• In Semester 2 (P3), adding hierarchical pooling (SAGPool/ASAP) + DropEdge/APPNP will let RGAT focus on salient depressive segments.",
        ],
        accent=TEAL,
        font_size=13.5,
    )

    # ── Slide 17: Confusion Matrices (Full-Width Plot) ────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Out-of-Fold Confusion Matrices (N = 189)", "Evaluating true depressed recall vs. false alarms without majority-class bias", 17)
    s.shapes.add_picture(str(RES_END / "confusion_matrices.png"), Inches(0.45), Inches(1.6), width=Inches(12.4))
    add_card(
        s, 0.6, 5.5, 12.1, 1.35,
        "Error Analysis",
        [
            "• SVM (RBF) predicts all 189 subjects as Control (0 depressed recall) — illustrating why accuracy alone (70.4%) is deceptive.",
            "• BiLSTM and RGAT GNN recover balanced sensitivity across both Control and Depressed classes via class-weighted BCE loss.",
        ],
        accent=NAVY,
        font_size=13.5,
    )

    # ── Slide 18: Gender Fairness Slice ───────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Performance Analysis: Gender Fairness Slice", "Auditing male vs. female UAR to motivate Semester 2 adversarial disentanglement", 18)
    f_rows = [[r.model, f"{r.female_uar:.3f}", f"{r.male_uar:.3f}", f"{r.gap_abs:.3f}"] for r in fairness_df.itertuples(index=False)]
    add_table(s, 0.6, 1.45, 7.1, 4.3, ["Model", "Female UAR (n=87)", "Male UAR (n=102)", "|UAR Gap|"], f_rows, highlight_rows={4, 5}, font_size=13.0)
    add_card(
        s, 7.95, 1.45, 4.75, 4.3,
        "Fairness Takeaways",
        [
            "• RGAT GNN gap narrowed to 2.6% (0.588 female vs 0.562 male) with Set A+C.",
            "• BiLSTM achieves 0.623 female / 0.616 male (0.7% gap).",
            "• Logistic Regression still exhibits a 12.5% gap (0.622 vs 0.497).",
            "• Semester 2 (P3.4): add a gradient-reversal gender adversarial head to learn gender-invariant graph embeddings.",
        ],
        accent=AMBER,
        font_size=13.5,
    )
    add_card(s, 0.6, 5.95, 12.1, 0.95, "Reproducibility", ["All tables and figures are reproduced deterministically by notebooks/02_end_review_pipeline.ipynb (Restart & Run All clean)."], accent=EMERALD, font_size=13.5)

    # ── Slide 19: Limitations & Future Scope ──────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "Limitations & Future Scope (Semester 2: Dec 2026 – Apr 2027)", "Semester 1 delivers P0 + P1 + P2; Semester 2 delivers P3 → P4 → P6 → P8", 19)
    add_card(
        s, 0.6, 1.45, 5.8, 5.3,
        "Current Limitations (Semester 1)",
        [
            "• Small dataset (189 subjects) → wide 95% CIs (±0.07).",
            "• Flat readout RGAT treats all ~221 segments equally (no hierarchical pooling or oversmoothing controls yet), trailing BiLSTM (0.589 vs 0.642).",
            "• No self-supervised speech embeddings (wav2vec2/HuBERT) yet.",
            "• Single-corpus evaluation (DAIC-WOZ only); cross-protocol generalization untested.",
            "• Set C uses human transcripts (not deployable on edge without ASR).",
        ],
        accent=AMBER,
        font_size=13.5,
    )
    add_card(
        s, 6.7, 1.45, 6.0, 5.3,
        "Future Scope — Semester 2 Roadmap",
        [
            "• P3 (Dec–Jan): Hierarchical pooling (SAGPool/ASAP), anti-oversmoothing (GIN/APPNP, DropEdge), contrastive pretraining (GraphCL), & gender adversarial head + ablation table.",
            "• P4 (Feb–Mar 2027): wav2vec2 stream + zero-shot cross-corpus eval on E-DAIC + segment-importance interpretability timelines.",
            "• P6 (Mar 2027): Final research paper + Docker reproducibility package.",
            "• P8 (Apr 2027): Set-A-only INT8 quantized GNN deployed & benchmarked on Raspberry Pi / Jetson Nano.",
        ],
        accent=EMERALD,
        font_size=13.5,
    )

    # ── Slide 20: References & Thank You ──────────────────────────────────────
    s = prs.slides.add_slide(blank)
    add_bg(s)
    add_header(s, "References & Acknowledgements", "Thank you — Questions & Discussion", 20)
    refs = [
        "[1] M. Valstar et al., \"AVEC 2016: Depression, Mood, and Emotion Recognition Workshop and Challenge,\" Proc. ACM Multimedia AVEC, 2016.",
        "[2] S. S. Leal, S. Ntalampiras, and R. Sassi, \"Speech-Based Depression Assessment: A Comprehensive Survey,\" IEEE Trans. Affective Computing, vol. 16, no. 3, 2025.",
        "[3] D. Macky Aleagha, P. Zohari, and M. Haghir Chehreghani, \"AI Models for Depressive Disorder Detection and Diagnosis: A Review,\" arXiv:2508.12022, 2025.",
        "[4] J. Gratch et al., \"The Distress Analysis Interview Corpus of Human and Computer Interviews,\" Proc. LREC, 2014.",
        "[5] G. Degottex et al., \"COVAREP: A Collaborative Voice Analysis Repository for Speech Technologies,\" Proc. IEEE ICASSP, 2014.",
        "[6] K. Kroenke et al., \"The PHQ-8 as a Measure of Current Depression in the General Population,\" J. Affective Disorders, vol. 114, 2009.",
    ]
    add_card(s, 0.6, 1.4, 12.1, 4.15, "References", refs, accent=NAVY, font_size=13.0)
    add_card(
        s, 0.6, 5.75, 12.1, 1.1,
        "Team & Supervision",
        ["Sahitya Kushwaha (623195)  ·  Buddhana Pallavi (623106)  ·  Potti Pothan Abhishek (623156)   |   Guide: Dr. V. P. Singh, ECE, NIT Andhra Pradesh"],
        accent=TEAL,
        font_size=13.5,
    )

    prs.save(OUT_PPT)
    print(f"Saved {OUT_PPT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_deck()
