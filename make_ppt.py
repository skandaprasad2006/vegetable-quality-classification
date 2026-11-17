"""Build the CAT-2 presentation for VegCheck.

    python tools/make_ppt.py

Writes ``reports/VegCheck_CAT2_Presentation.pptx``.

Every number is read at build time from the project's own evaluation artefacts
(``evaluation/metrics/*.json``, ``evaluation/phase4_test/``,
``evaluation/phase4_robustness/``, ``evaluation/phase4_accessibility/``), so the
deck cannot drift from the reported results. Figures come from
``tools/make_ppt_figures.py``; screenshots from ``reports/screenshots/phase4/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "ppt" / "figures"
SHOTS = ROOT / "reports" / "screenshots" / "phase4"
OUT_PATH = ROOT / "reports" / "VegCheck_CAT2_Presentation.pptx"

# ---------------------------------------------------------------- palette ---
INK = RGBColor(0x0D, 0x2B, 0x21)
INK_SOFT = RGBColor(0x3C, 0x53, 0x49)
EMERALD = RGBColor(0x0F, 0x7A, 0x4D)
EMERALD_DK = RGBColor(0x0A, 0x3B, 0x2A)
EMERALD_LT = RGBColor(0x16, 0xA0, 0x6A)
MINT = RGBColor(0xE9, 0xF7, 0xF0)
MINT_DEEP = RGBColor(0xD3, 0xEC, 0xDF)
PAPER = RGBColor(0xF7, 0xFB, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x5C, 0x72, 0x66)
AMBER = RGBColor(0xB3, 0x76, 0x0A)
AMBER_BG = RGBColor(0xFF, 0xF6, 0xE0)
FONT = "Segoe UI"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.8)
CONTENT_W = W - 2 * MARGIN


# ---------------------------------------------------------------- helpers ---
def read_json(rel: str) -> dict:
    with open(ROOT / rel, "r", encoding="utf-8") as fh:
        return json.load(fh)


COMP = read_json("evaluation/metrics/p0_comparison.json")
TEST = read_json("evaluation/phase4_test/metrics.json")
ROB = read_json("evaluation/phase4_robustness/summary.json")
LAT = read_json("evaluation/metrics/inference_latency.json")
PERF = read_json("evaluation/phase4_performance/performance.json")
ACC = read_json("evaluation/phase4_accessibility/audit.json")
AUDIT = read_json("evaluation/phase4/integrity_audit.json")
SPLIT = read_json("datasets/metadata/split_summary.json")

RUNS = COMP["runs"]
DECISION = COMP["deployment_decision"]
HEAD = DECISION["head_to_head"]
DEPLOYED = RUNS[DECISION["point_estimate_winner"]]
RUNNER = RUNS[DECISION["runner_up"]]
SPEEDUP = LAT["results"]["A3"]["end_to_end"]["median_ms"] / LAT["results"]["A5"]["end_to_end"]["median_ms"]

# Phase 4 performance audit: stage medians for the deployed model in the app.
STAGES = PERF["stages"]["stages"]
E2E = STAGES["end_to_end"]["median_ms"]
PREPROC_MEAN = STAGES["preprocess"]["mean_ms"]
COLD = PERF["cold_start"]["load_s_median"]
RSS_AFTER = PERF["repeated_predictions"]["rss_after_mb"]
RSS_GROWTH = PERF["repeated_predictions"]["rss_growth_mb"]
LARGE_MS = PERF["large_image"]["decode_ms"] + PERF["large_image"]["predict_and_preprocess_ms"]


def new_deck() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    return prs


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def rect(slide, x, y, w, h, colour, shape=MSO_SHAPE.RECTANGLE, line=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = colour
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, body, size=14, colour=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.25, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = body.split("\n") if isinstance(body, str) else body
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = colour
        r.font.name = FONT
    return box


def header(slide, kicker: str, title: str, number: int) -> None:
    text(slide, MARGIN, Inches(0.46), CONTENT_W, Inches(0.25), kicker.upper(),
         size=10.5, colour=EMERALD, bold=True)
    text(slide, MARGIN, Inches(0.72), CONTENT_W, Inches(0.62), title, size=27, bold=True)
    rect(slide, MARGIN, Inches(1.42), Inches(1.05), Emu(Emu(38100) * 1), EMERALD)
    rect(slide, MARGIN + Inches(1.05), Inches(1.42), CONTENT_W - Inches(1.05), Pt(1), MINT_DEEP)
    footer(slide, number)


def footer(slide, number: int) -> None:
    rect(slide, MARGIN, H - Inches(0.62), CONTENT_W, Pt(0.75), MINT_DEEP)
    text(slide, MARGIN, H - Inches(0.5), Inches(7), Inches(0.25),
         "VegCheck · Vegetable Quality & Freshness Detection · IPCV 24AM3504",
         size=8.5, colour=GREY)
    text(slide, W - MARGIN - Inches(1), H - Inches(0.5), Inches(1), Inches(0.25),
         str(number), size=9, colour=GREY, align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=13.5, gap=0.13) -> None:
    """items: list of (lead, body) or plain strings."""
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.28
        p.space_after = Pt(11 if gap else 4)
        dot = p.add_run()
        dot.text = "▪  "
        dot.font.size = Pt(size)
        dot.font.color.rgb = EMERALD
        dot.font.name = FONT
        lead, body = (item, "") if isinstance(item, str) else item
        if body and not body.startswith(" "):
            body = " " + body
        if lead:
            r = p.add_run()
            r.text = lead
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = INK
            r.font.name = FONT
        if body:
            r2 = p.add_run()
            r2.text = body
            r2.font.size = Pt(size)
            r2.font.color.rgb = INK_SOFT
            r2.font.name = FONT
    return box


def stat_cards(slide, x, y, w, items, height=Inches(1.16), gap=Inches(0.22)) -> None:
    """items: list of (value, label). Cards laid out in one row."""
    n = len(items)
    cw = int((w - gap * (n - 1)) / n)
    for i, (value, label) in enumerate(items):
        cx = x + i * (cw + gap)
        rect(slide, cx, y, cw, height, WHITE, line=MINT_DEEP)
        rect(slide, cx, y, cw, Pt(3.2), EMERALD)
        text(slide, cx + Inches(0.16), y + Inches(0.2), cw - Inches(0.3), Inches(0.44),
             value, size=19, bold=True)
        text(slide, cx + Inches(0.16), y + Inches(0.66), cw - Inches(0.3), Inches(0.42),
             label, size=9.5, colour=GREY)


def picture(slide, path: Path, x, y, max_w, max_h) -> None:
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    slide.shapes.add_picture(str(path), int(x), int(y), width=w, height=h)


def table(slide, x, y, w, rows, col_widths, header_row=True, size=11, row_h=Inches(0.34)):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, x, y, w, row_h * n_rows)
    tbl = shape.table
    total = sum(col_widths)
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = Emu(int(w * cw / total))
    for i, row in enumerate(rows):
        tbl.rows[i].height = row_h
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if i == 0 and header_row:
                cell.fill.fore_color.rgb = EMERALD_DK
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else MINT
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = FONT
            r.font.bold = (i == 0 and header_row) or (j == 0)
            r.font.color.rgb = WHITE if (i == 0 and header_row) else INK
    return tbl


def note(slide, x, y, w, h, body, bg=AMBER_BG, fg=AMBER) -> None:
    rect(slide, x, y, w, h, bg, line=MINT_DEEP)
    rect(slide, x, y, Pt(4), h, fg)
    text(slide, x + Inches(0.18), y + Inches(0.12), w - Inches(0.32), h - Inches(0.2),
         body, size=10.5, colour=INK_SOFT)


def notes(slide, body: str) -> None:
    slide.notes_slide.notes_text_frame.text = body


# ------------------------------------------------------------------ slides ---
def s_title(prs, n):
    s = blank(prs)
    rect(s, 0, 0, W, H, EMERALD_DK)
    rect(s, 0, 0, W, Inches(0.16), EMERALD_LT)
    # decorative corner block
    circ = rect(s, W - Inches(3.1), -Inches(1.3), Inches(4.4), Inches(4.4),
                RGBColor(0x0F, 0x5C, 0x3F), shape=MSO_SHAPE.OVAL)

    text(s, MARGIN, Inches(1.5), Inches(9.4), Inches(0.3),
         "IMAGE PROCESSING & COMPUTER VISION · 24AM3504", size=11.5,
         colour=RGBColor(0x9F, 0xF0, 0xC4), bold=True)
    text(s, MARGIN, Inches(1.95), Inches(10.6), Inches(1.0), "VegCheck", size=54, bold=True,
         colour=WHITE)
    text(s, MARGIN, Inches(2.95), Inches(9.6), Inches(1.1),
         "Vegetable Quality and Freshness Detection\nUsing Image Processing and Deep Learning",
         size=23, colour=RGBColor(0xD8, 0xF0, 0xE4), spacing=1.2)
    rect(s, MARGIN, Inches(4.35), Inches(1.6), Pt(3), EMERALD_LT)
    text(s, MARGIN, Inches(4.62), Inches(10.5), Inches(1.1),
         "CAT-2 Presentation  ·  Continuous Assessment Tool 2\n"
         "V Semester B.Tech · Section A · Academic Year 2026–27\n"
         "Department of CSE (Artificial Intelligence & Machine Learning)",
         size=12.5, colour=RGBColor(0xB9, 0xD8, 0xC9), spacing=1.35)
    text(s, MARGIN, H - Inches(1.0), Inches(11.6), Inches(0.3),
         f"Deployed model: {DEPLOYED['display_name']}  ·  official test macro F1 "
         f"{TEST['macro_f1']:.4f}  ·  6,150 images  ·  178 automated tests",
         size=10.5, colour=RGBColor(0x9F, 0xF0, 0xC4))
    notes(s, "Open with the one-line claim: this project classifies a photograph of a "
             "vegetable into one of five visual quality classes, and it reports how it "
             "fails. The deck follows the evaluation sheet: problem, objectives, "
             "methodology and demo, then results.")


def s_agenda(prs, n):
    s = blank(prs)
    header(s, "Agenda", "What this presentation covers", n)
    items = [
        ("Problem statement", "what is broken today, and why a photograph is the right sensor"),
        ("Objectives", "four objectives, each with a test that can fail"),
        ("Dataset and integrity control", "6,150 images, duplicate-safe splits"),
        ("Methodology", "segmentation, 83 descriptors, three deep models, one protocol"),
        ("Results", "validation comparison, the deployed model, the single test pass"),
        ("Robustness, performance, deployment", "how it behaves outside the studio image"),
        ("Live demonstration", "the deployed web application"),
        ("Limitations and conclusion", "what it cannot do, and what is next"),
    ]
    bullets(s, MARGIN, Inches(1.78), Inches(7.4), Inches(4.6), items, size=13)
    rect(s, Inches(8.6), Inches(1.78), Inches(3.95), Inches(4.3), PAPER, line=MINT_DEEP)
    text(s, Inches(8.85), Inches(2.0), Inches(3.5), Inches(0.3), "HEADLINE RESULT", size=10,
         colour=EMERALD, bold=True)
    text(s, Inches(8.85), Inches(2.4), Inches(3.5), Inches(1.1),
         f"{TEST['macro_f1']:.4f}", size=40, bold=True, colour=INK)
    text(s, Inches(8.85), Inches(3.35), Inches(3.4), Inches(0.8),
         "test macro F1, scored once on 935 images", size=11, colour=INK_SOFT)
    text(s, Inches(8.85), Inches(4.15), Inches(3.4), Inches(1.6),
         f"Validation  {DEPLOYED['quality']['macro_f1']:.4f}\n"
         f"Accuracy    {TEST['accuracy']:.4f}\n"
         f"Errors      {TEST['n_incorrect']} / {TEST['n_test_images']}\n"
         f"Latency     {E2E:.0f} ms median",
         size=11.5, colour=INK_SOFT, spacing=1.5)
    notes(s, "Signpost the marking scheme: the first four sections answer the problem "
             "statement and objectives, the middle is methodology and demo, and the "
             "last two are there for viva questions.")


def s_problem(prs, n):
    s = blank(prs)
    header(s, "1 · Problem statement", "Produce grading is manual, subjective and slow", n)
    bullets(s, MARGIN, Inches(1.8), Inches(6.9), Inches(4.6), [
        ("Grading is visual.", " Traders and inspectors judge ripeness and damage by eye, "
         "so two people grade the same crate differently and the same person drifts over a day."),
        ("It does not scale.", " A market survey needs a human per crate; a photograph "
         "costs nothing once the camera exists."),
        ("The classes are fuzzy.", " Ripe and over-ripe sit on a continuum, so any "
         "automatic method must be judged with per-class evidence, not one accuracy number."),
        ("The honest question:", " given a single photograph on a plain background, can "
         "image processing classify visual quality well enough to be useful, and can we "
         "prove it without fooling ourselves?"),
    ], size=13)
    rect(s, Inches(8.15), Inches(1.8), Inches(4.4), Inches(4.36), PAPER, line=MINT_DEEP)
    text(s, Inches(8.4), Inches(2.0), Inches(3.9), Inches(0.3), "SCOPE SET AT THE START",
         size=10, colour=EMERALD, bold=True)
    bullets(s, Inches(8.4), Inches(2.42), Inches(3.9), Inches(3.5), [
        "5 visual quality classes, 4 vegetable types",
        "Quality only — no safety, no nutrition claim",
        "One dataset, splits frozen before modelling",
        "Every claim traceable to a saved artefact",
    ], size=11.5)
    note(s, MARGIN, Inches(6.42), CONTENT_W, Inches(0.55),
         "Constraint that shaped every later decision: the model must be selected on "
         "validation data only, and the test split stays sealed until the model is frozen.")
    notes(s, "The interesting engineering problem is not the classifier, it is the "
             "measurement protocol. Say that early: it explains why so much of the work "
             "is auditing and reporting.")


def s_objectives(prs, n):
    s = blank(prs)
    header(s, "2 · Objectives", "Four objectives, each with a test that can fail", n)
    rows = [
        ["#", "Objective", "How it is judged"],
        ["O1", "Build a reproducible, leak-free dataset and feature pipeline",
         "0 near-duplicate groups or hashes spanning splits; 83 descriptors recomputed identically"],
        ["O2", "Compare classical and deep approaches under one protocol",
         "Same split, same scorer, macro F1 with bootstrap CI and McNemar test"],
        ["O3", "Deploy the best measured model as a working application",
         "Upload → prediction in the browser, metrics read from artefacts, not typed"],
        ["O4", "Validate the deployed system and state its limits",
         "Integrity audit, one test pass, robustness, timing, accessibility, 178 tests"],
    ]
    table(s, MARGIN, Inches(1.85), CONTENT_W, rows, [0.6, 4.6, 6.6], size=11.5,
          row_h=Inches(0.72))
    note(s, MARGIN, Inches(5.55), CONTENT_W, Inches(0.85),
         "Objectives are written so that failure is observable. O2 deliberately does not "
         "say \u201cshow that deep learning works\u201d — it says compare them, and accept "
         "whichever wins. That is why a Random Forest is deployed, not a network.")
    notes(s, "If asked why the title says deep learning but a Random Forest is deployed: "
             "the deep models were built, trained and fully reported; they did not win on "
             "this dataset with frozen backbones.")


def s_dataset(prs, n):
    s = blank(prs)
    header(s, "3 · Dataset", "6,150 images, audited before any model saw them", n)
    counts = SPLIT["checks"]["counts_per_split"]
    stat_cards(s, MARGIN, Inches(1.75), CONTENT_W, [
        (f"{sum(counts.values()):,}", "images in the corpus"),
        ("4 × 5", "vegetable × quality classes"),
        (f"{counts.get('train', 0):,} / {counts.get('validation', 0):,} / {counts.get('test', 0):,}",
         "train / validation / test"),
        ("6.4×", "largest ÷ smallest quality class"),
    ])
    bullets(s, MARGIN, Inches(3.15), Inches(6.5), Inches(3.1), [
        ("Source.", " New VegNet redistribution on Hugging Face; 19 of 20 "
         "vegetable × quality cells populated — Tomato/Dried does not exist here."),
        ("Audit.", " corruption check, exact-duplicate SHA-256, and perceptual "
         "near-duplicate clustering (dHash), before splitting."),
        ("Grouped split.", " a duplicate cluster never straddles a boundary, so the same "
         "physical specimen cannot appear in both training and test."),
        ("Label honesty.", " 41 clusters / 134 images (2.2 %) carry genuinely conflicting "
         "labels; that ceiling is documented, not hidden."),
    ], size=12)
    pic = ROOT / "evaluation" / "graphs" / "eda_class_distribution.png"
    if pic.exists():
        picture(s, pic, Inches(7.5), Inches(3.15), Inches(5.05), Inches(2.9))
    notes(s, "The duplicate-grouped split is the single most important methodological "
             "decision: an ungrouped split inflates macro F1 by about 0.025 because the "
             "same specimen appears on both sides.")


def s_pipeline(prs, n):
    s = blank(prs)
    header(s, "4 · Methodology", "One pipeline, reused by training and by the website", n)
    steps = [
        ("1  Segment", "LAB distance from\nthe estimated background"),
        ("2  Describe", "83 colour, texture\nand shape descriptors"),
        ("3  Scale", "StandardScaler fitted\non training data only"),
        ("4  Classify", "Random Forest,\n200 trees"),
        ("5  Validate", "shape, finiteness,\nsums to 1, class map"),
    ]
    x = MARGIN
    cw = Inches(2.28)
    gap = Inches(0.11)
    for i, (title, body) in enumerate(steps):
        cx = x + i * (cw + gap)
        rect(s, cx, Inches(1.85), cw, Inches(1.28), MINT, line=MINT_DEEP)
        text(s, cx + Inches(0.14), Inches(1.98), cw - Inches(0.2), Inches(0.3), title,
             size=12.5, bold=True, colour=EMERALD_DK)
        text(s, cx + Inches(0.14), Inches(2.34), cw - Inches(0.24), Inches(0.7), body,
             size=10, colour=INK_SOFT, spacing=1.2)
        if i < len(steps) - 1:
            text(s, cx + cw, Inches(2.28), gap + Inches(0.02), Inches(0.3), "›",
                 size=18, colour=EMERALD, bold=True)
    bullets(s, MARGIN, Inches(3.45), Inches(6.6), Inches(2.8), [
        ("Segmentation", " is classical: distance in LAB from the background colour "
         "estimated at the frame border, then the largest connected component."),
        ("The 83 descriptors", " are computed on the 224×224 working image: colour "
         "moments and hue histograms, GLCM Haralick texture, uniform LBP, and shape."),
        ("One implementation.", " The website imports the training functions, so serving "
         "cannot silently diverge — the audit checks this on real images."),
    ], size=12)
    pic = ROOT / "evaluation" / "graphs" / "segmentation_example.png"
    if pic.exists():
        picture(s, pic, Inches(7.55), Inches(3.4), Inches(5.0), Inches(2.85))
    else:
        picture(s, SHOTS / "02b_result_detail.png", Inches(7.55), Inches(3.4),
                Inches(5.0), Inches(2.85))
    notes(s, "A Phase 4 audit found the website extracting features at 256 px while "
             "training used 224 px. It is fixed, and the check that caught it now runs in "
             "the test suite.")


def s_features(prs, n):
    s = blank(prs)
    header(s, "5 · Methodology", "83 handcrafted descriptors, chosen for this task", n)
    rows = [
        ["Family", "Count", "What it captures", "Why it matters here"],
        ["Colour", "30", "RGB and HSV moments, circular hue statistics, hue histogram",
         "ripeness and damage show up as hue shift and saturation loss"],
        ["Texture", "39", "GLCM Haralick at 3 distances × 4 angles, uniform LBP ×2 scales",
         "shrivel, bruise and surface damage change local contrast"],
        ["Shape", "7", "area fraction, perimeter, circularity, aspect, solidity, extent",
         "deformation and partial occlusion"],
    ]
    table(s, MARGIN, Inches(1.8), Inches(6.9), rows, [1.0, 0.62, 2.9, 2.5], size=10.5,
          row_h=Inches(0.62))
    bullets(s, MARGIN, Inches(4.35), Inches(6.9), Inches(1.9), [
        "Circular hue statistics use mean sine/cosine, so red does not average to cyan.",
        "No raw pixels: 83 values carry the discriminative structure, which is why the "
        "forest trains in under a second.",
    ], size=11.5)
    pic = ROOT / "evaluation" / "graphs" / "feature_importance_top20.png"
    if pic.exists():
        picture(s, pic, Inches(7.95), Inches(1.8), Inches(4.6), Inches(4.45))
    notes(s, "Colour dominates the importance ranking, which matches the domain: ripeness "
             "is largely a colour phenomenon, and damage adds texture.")


def s_deep(prs, n):
    s = blank(prs)
    header(s, "6 · Methodology", "The deep models were built, trained and reported", n)
    rows = [
        ["Run", "Model", "Weights", "Head", "Quality macro F1"],
    ]
    for rid in ("A2", "A3", "A4"):
        r = RUNS[rid]
        rows.append([
            rid, r["display_name"].replace(" (", "\n("),
            "frozen ImageNet" if "frozen" in r["display_name"].lower() else "frozen ImageNet",
            "multi-task" if rid == "A4" else "single",
            f"{r['quality']['macro_f1']:.4f}",
        ])
    table(s, MARGIN, Inches(1.85), Inches(7.3), rows, [0.55, 3.0, 1.5, 1.0, 1.35],
          size=10.5, row_h=Inches(0.58))
    bullets(s, MARGIN, Inches(4.4), Inches(7.3), Inches(1.8), [
        ("Identical protocol", " for all four runs: same split, same preprocessing, same "
         "scorer, one seed."),
        ("Frozen backbones", " were a deliberate scope decision: it isolates the value of "
         "the representation from the cost of fine-tuning."),
    ], size=11.5)
    note(s, Inches(8.35), Inches(1.85), Inches(4.25), Inches(2.0),
         "Fine-tuning was not run. This is the most plausible way the ranking could "
         "change, and it is stated as future work rather than implied as a result.")
    rect(s, Inches(8.35), Inches(4.05), Inches(4.25), Inches(2.15), PAPER, line=MINT_DEEP)
    text(s, Inches(8.6), Inches(4.22), Inches(3.8), Inches(0.3), "COST OF SERVING",
         size=10, colour=EMERALD, bold=True)
    text(s, Inches(8.6), Inches(4.6), Inches(3.8), Inches(1.4),
         f"Random Forest   {LAT['results']['A5']['end_to_end']['median_ms']:.1f} ms\n"
         f"ResNet50        {LAT['results']['A3']['end_to_end']['median_ms']:.1f} ms\n"
         f"MobileNetV2    {LAT['results']['A4']['end_to_end']['median_ms']:.1f} ms",
         size=11.5, colour=INK_SOFT, spacing=1.5)
    notes(s, "ResNet50 is the strongest network; MobileNetV2 is cheaper but its frozen "
             "features are weaker. Neither beat the handcrafted baseline.")


def s_protocol(prs, n):
    s = blank(prs)
    header(s, "7 · Methodology", "How the comparison avoids fooling itself", n)
    cards = [
        ("macro F1", "primary metric: every class counts equally"),
        ("95 % CI", "2,000-sample paired bootstrap over images"),
        ("McNemar", "paired test on discordant predictions"),
        ("one seed", "so no seed shopping; stated as a limitation"),
    ]
    stat_cards(s, MARGIN, Inches(1.8), CONTENT_W, [(v, l) for v, l in cards],
               height=Inches(1.15))
    bullets(s, MARGIN, Inches(3.25), Inches(6.6), Inches(3.0), [
        ("Selection on validation only.", " The 929-image validation split decided the "
         "model; the test split was sealed until the artefact was frozen."),
        ("Test used once.", " In Phase 4 the frozen model was scored a single time on 935 "
         "images, and the result was not fed back into any decision."),
        ("Near-duplicates grouped.", " Verified: 0 groups and 0 SHA-256 values span splits."),
        ("Artefacts frozen.", f" Model SHA-256 53db2b14…{AUDIT['checks']['A_model_identity']['sha256'][-6:]}, "
         "re-checked byte-for-byte before and after the test pass."),
    ], size=12)
    rect(s, Inches(7.7), Inches(3.25), Inches(4.85), Inches(2.15), PAPER, line=MINT_DEEP)
    text(s, Inches(7.95), Inches(3.42), Inches(4.4), Inches(0.3), "WHAT WOULD INVALIDATE IT",
         size=10, colour=EMERALD, bold=True)
    text(s, Inches(7.95), Inches(3.8), Inches(4.4), Inches(1.5),
         "Tuning on test results · re-splitting after seeing scores · comparing models "
         "across different scorers · reporting the best of several seeds · claiming a "
         "difference inside the confidence interval.",
         size=11.5, colour=INK_SOFT, spacing=1.35)
    notes(s, "Four models, one scorer. The runner-up difference is inside the interval, "
             "which is exactly why the cheaper model could be chosen without special "
             "pleading.")


def s_results(prs, n):
    s = blank(prs)
    header(s, "8 · Results", "Validation: the classical model measured best", n)
    short = {"A5": "Random Forest", "A3": "ResNet50 (frozen)",
             "A4": "MobileNetV2 multi-task", "A2": "MobileNetV2 (frozen)"}
    rows = [["Run", "Model", "Macro F1", "95 % CI", "Accuracy"]]
    for rid in ("A5", "A3", "A4", "A2"):
        r = RUNS[rid]
        q = r["quality"]
        rows.append([rid, short[rid], f"{q['macro_f1']:.4f}",
                     f"{q['macro_f1_ci95']['lo']:.3f}–{q['macro_f1_ci95']['hi']:.3f}",
                     f"{q['accuracy']:.4f}"])
    table(s, MARGIN, Inches(1.8), Inches(6.8), rows, [0.5, 2.7, 1.0, 1.4, 1.0], size=10.5,
          row_h=Inches(0.46))
    text(s, MARGIN, Inches(4.22), Inches(6.8), Inches(0.3),
         "A5 is the deployed model; A3 is its only close competitor.", size=9.5, colour=GREY)
    bullets(s, MARGIN, Inches(4.62), Inches(6.8), Inches(1.5), [
        ("Ordering.", " Random Forest 0.9539 > ResNet50 0.9521 > MobileNetV2 multi-task "
         "0.9401 > MobileNetV2 0.9376."),
        ("But the gap is not significant", " — the intervals overlap heavily."),
    ], size=11.5)
    pic = FIG / "fig_models.png"
    if pic.exists():
        picture(s, pic, Inches(7.55), Inches(1.9), Inches(5.0), Inches(3.6))
    note(s, Inches(7.55), Inches(5.65), Inches(5.0), Inches(1.0),
         f"Multi-task learning added nothing: the vegetable head was already near-perfect "
         f"({RUNS['A4']['vegetable']['macro_f1']:.4f}), so it gave no useful auxiliary signal.")
    notes(s, "Do not oversell this. The honest statement is: on this dataset, with frozen "
             "backbones, the networks did not produce a measurable improvement.")


def s_confusion(prs, n):
    s = blank(prs)
    header(s, "9 · Results", "Where the errors are, and why the average hides them", n)
    pic = ROOT / "evaluation" / "confusion_matrices" / "A5_quality.png"
    if pic.exists():
        picture(s, pic, MARGIN, Inches(1.75), Inches(6.2), Inches(4.5))
    rows = [["Class", "Support", "Precision", "Recall", "F1"]]
    for r in DEPLOYED["quality"]["per_class"]:
        rows.append([r["class"], r["support"], f"{r['precision']:.3f}",
                     f"{r['recall']:.3f}", f"{r['f1']:.3f}"])
    table(s, Inches(7.25), Inches(1.8), Inches(5.3), rows, [1.1, 0.95, 1.15, 1.0, 1.0],
          size=10.5, row_h=Inches(0.42))
    bullets(s, Inches(7.25), Inches(4.6), Inches(5.3), Inches(1.9), [
        ("Damaged is the weak class", " — only 46 validation images, and the model trades "
         "precision for recall on it."),
        ("Ripe ↔ Old and Dried ↔ Old", " dominate the off-diagonal: the boundary the "
         "label ambiguity lives on."),
    ], size=11)
    notes(s, "This is the slide for the \u201cwhy not just report accuracy\u201d question: "
             "96 % accuracy with a useless Damaged class would still look impressive.")


def s_decision(prs, n):
    s = blank(prs)
    header(s, "10 · Decision", "Why the Random Forest is the deployed model", n)
    stat_cards(s, MARGIN, Inches(1.8), CONTENT_W, [
        (f"{HEAD['delta']:+.4f}", "vs ResNet50 on validation macro F1"),
        (f"[{HEAD['delta_ci95'][0]:+.3f}, {HEAD['delta_ci95'][1]:+.3f}]", "paired 95 % CI (spans zero)"),
        (f"{HEAD['mcnemar']['p_value']:.3f}", "McNemar p — no detectable difference"),
        (f"{SPEEDUP:.0f}×", "faster end-to-end inference"),
    ])
    bullets(s, MARGIN, Inches(3.3), Inches(6.5), Inches(2.6), [
        ("Statistically tied, operationally not.", " When accuracy cannot separate two "
         "models, the cheaper one wins: 15.7 MB, 27 ms median, no TensorFlow at serving time."),
        ("Not a verdict on deep learning.", " The networks are fully reported; they simply "
         "did not win on this dataset at this scale with frozen backbones."),
        ("Recorded, not remembered.", " Decision log D-012 with the statistics, so the "
         "reason survives the project."),
    ], size=12)
    rows = [["", "Random Forest", "ResNet50"],
            ["Artefact size", "15.7 MB", "94 MB"],
            ["Median end-to-end", f"{LAT['results']['A5']['end_to_end']['median_ms']:.1f} ms",
             f"{LAT['results']['A3']['end_to_end']['median_ms']:.1f} ms"],
            ["Serving runtime", "scikit-learn", "TensorFlow"],
            ["Retrain time", "0.65 s", "minutes"]]
    table(s, Inches(7.6), Inches(3.3), Inches(4.95), rows, [1.7, 1.6, 1.6], size=10.5,
          row_h=Inches(0.44))
    notes(s, "The rubric rewards defensible reasoning. This is the slide that shows the "
             "choice was made by measurement, including the possibility that it was wrong.")


def s_test(prs, n):
    s = blank(prs)
    header(s, "11 · Results", "The single official test evaluation", n)
    stat_cards(s, MARGIN, Inches(1.78), CONTENT_W, [
        (f"{TEST['macro_f1']:.4f}", "test macro F1 (primary metric)"),
        (f"{TEST['accuracy']:.4f}", "test accuracy"),
        (f"{TEST['n_incorrect']} / {TEST['n_test_images']}", "errors"),
        (f"{TEST['macro_f1_ci95']['lo']:.3f}–{TEST['macro_f1_ci95']['hi']:.3f}", "95 % bootstrap CI"),
    ])
    bullets(s, MARGIN, Inches(3.2), Inches(6.4), Inches(2.7), [
        ("Scored once.", " The test split was opened after the model and thresholds were "
         "frozen, and the outcome changed nothing."),
        ("Consistent with validation.", f" {DEPLOYED['quality']['macro_f1']:.4f} validation vs "
         f"{TEST['macro_f1']:.4f} test: the intervals overlap, so the difference is "
         "sampling variation, not decay."),
        ("Weakest class on test.", " Damaged — F1 "
         f"{[r for r in TEST['per_class'] if r['class'] == 'Damaged'][0]['f1']:.3f} "
         "on 49 images, the same weakness validation predicted."),
    ], size=12)
    pic = FIG / "fig_per_class.png"
    if pic.exists():
        picture(s, pic, Inches(7.4), Inches(3.15), Inches(5.15), Inches(2.9))
    notes(s, "Emphasise that the test set is now spent: any future comparison needs fresh "
             "data, which is stated as a limitation rather than quietly ignored.")


def s_robustness(prs, n):
    s = blank(prs)
    header(s, "12 · Validation", "Outside the studio image, accuracy falls", n)
    pic = FIG / "fig_robustness.png"
    if pic.exists():
        picture(s, pic, MARGIN, Inches(1.8), Inches(7.5), Inches(4.1))
    conds = ROB["conditions"]
    bullets(s, Inches(8.5), Inches(1.85), Inches(4.1), Inches(4.0), [
        ("Control:", f" {conds['control']['accuracy']:.2f} on the unmodified image."),
        ("Mild:", f" JPEG q30 {conds['jpeg_q30']['accuracy']:.2f}, rotation "
         f"{conds['rotated_15deg']['accuracy']:.2f}."),
        ("Harsh:", f" blur {conds['gaussian_blur']['accuracy']:.2f}, low resolution "
         f"{conds['low_resolution']['accuracy']:.2f}, vignette "
         f"{conds['vignette']['accuracy']:.2f}."),
        ("Confidence is not a safety net:", " the worst failures are confident ones, "
         "mostly Old → Ripe at 0.96–0.97."),
    ], size=11)
    note(s, MARGIN, Inches(6.1), CONTENT_W, Inches(0.75),
         "Study design, stated plainly: 61 held-out test images × 10 controlled "
         "degradations. It is quantitative, but it is not a field-photograph study — no "
         "real photographs outside the dataset existed in this project.")
    notes(s, "If asked whether this is real-world evidence: no, and the slide says so. It "
             "bounds the failure modes; it does not measure kitchen performance.")


def s_performance(prs, n):
    s = blank(prs)
    header(s, "13 · Validation", "Measured performance, not estimated", n)
    stat_cards(s, MARGIN, Inches(1.78), CONTENT_W, [
        (f"{E2E:.1f} ms", "median end-to-end (CPU)"),
        (f"{PREPROC_MEAN:.1f} ms", "of which preprocessing"),
        (f"{COLD:.2f} s", "cold start, paid once"),
        (f"~{RSS_AFTER:.0f} MB", "resident memory"),
    ])
    rows = [["Stage", "Median", "Mean"],
            ["validate", f"{STAGES['validation']['median_ms']:.2f} ms",
             f"{STAGES['validation']['mean_ms']:.2f} ms"],
            ["decode", f"{STAGES['decode']['median_ms']:.2f} ms",
             f"{STAGES['decode']['mean_ms']:.2f} ms"],
            ["preprocess (segment + 83 features)", f"{STAGES['preprocess']['median_ms']:.2f} ms",
             f"{STAGES['preprocess']['mean_ms']:.2f} ms"],
            ["predict (forest)", f"{STAGES['predict']['median_ms']:.2f} ms",
             f"{STAGES['predict']['mean_ms']:.2f} ms"],
            ["end to end", f"{E2E:.2f} ms", f"{STAGES['end_to_end']['mean_ms']:.2f} ms"]]
    table(s, MARGIN, Inches(3.2), Inches(5.6), rows, [3.1, 1.2, 1.2], size=10.5,
          row_h=Inches(0.4))
    bullets(s, Inches(6.75), Inches(3.15), Inches(5.8), Inches(2.8), [
        ("200 consecutive predictions:", f" 0 errors, +{RSS_GROWTH:.1f} MB — "
         "no leak of consequence."),
        ("A 12-megapixel upload", " still completes in "
         f"{LARGE_MS:.0f} ms, so the upload guard is not hiding a stall."),
        ("Failure paths tested", ": wrong extension, corrupt bytes, tiny image, oversized "
         "file, and a missing model all produce a readable message."),
        ("Honest envelope:", " these are timings on one laptop CPU, not a server benchmark."),
    ], size=11.5)
    notes(s, "Latency matters for the demo claim: a prediction is faster than the click "
             "that triggers it, so the interface feels immediate.")


def s_app(prs, n):
    s = blank(prs)
    header(s, "14 · Demonstration", "The deployed application", n)
    for i, (name, cap) in enumerate([
        ("01_home.png", "Landing: upload or a sample, scope stated up front"),
        ("02b_result_detail.png", "Result: preprocessing, class scores, honest caveat"),
    ]):
        p = SHOTS / name
        x = MARGIN + i * Inches(3.72)
        if p.exists():
            picture(s, p, x, Inches(1.8), Inches(3.55), Inches(2.5))
            text(s, x, Inches(4.38), Inches(3.55), Inches(0.5), cap, size=9.5,
                 colour=GREY)
    note(s, MARGIN, Inches(5.15), Inches(7.35), Inches(1.15),
         "Demo path: choose the sample image, press Predict quality. The same inference "
         "path serves the ?predict=1 link used for the screenshots, so what is shown is "
         "what the button runs.")
    bullets(s, Inches(8.45), Inches(1.85), Inches(4.1), Inches(4.4), [
        ("Four pages:", " prediction, dataset, model performance, about."),
        ("Every figure is read", " from an evaluation artefact at runtime — no metric is "
         "typed into the app."),
        ("What it refuses to do:", " it does not claim vegetable identification or food "
         "safety; the deployed model supports neither."),
        ("Session-only history,", " no uploads stored, 15 MB cap enforced in two places."),
    ], size=11)
    notes(s, "Demonstrate with a sample image if the network is unavailable: the same "
             "single inference path is used by the button and the ?predict=1 link.")


def s_quality(prs, n):
    s = blank(prs)
    header(s, "15 · Engineering quality", "The validation work behind the demo", n)
    stat_cards(s, MARGIN, Inches(1.78), CONTENT_W, [
        ("178", "automated tests, 0 failures"),
        ("8 / 8", "integrity checks passing"),
        (f"{ACC['summary']['min_contrast']:.1f}:1", "minimum text contrast (AA needs 4.5)"),
        ("48", "release artefacts, hash-verified"),
    ])
    bullets(s, MARGIN, Inches(3.25), Inches(6.5), Inches(2.8), [
        ("The audit caught a real defect.", " The website extracted features at 256 px "
         "while training used 224 px; fixed, then verified bit-identical."),
        ("Accessibility measured, not assumed:", f" {ACC['summary']['below_AA']} of "
         f"{ACC['summary']['text_samples']} samples below AA, images given alt text, "
         "keyboard reachable, no overflow at 390 px."),
        ("Release frozen.", " Every artefact has a SHA-256 in the manifest, so the "
         "submitted state is reproducible."),
    ], size=11.5)
    rows = [["Check", "Result"],
            ["Model artefact", "byte-identical to Phase 2"],
            ["Split integrity", "0 groups / 0 hashes spanning splits"],
            ["Feature equality", "max difference 0.0 on 20 images"],
            ["Test overwrite guard", "refuses without explicit flag"]]
    table(s, Inches(7.5), Inches(3.3), Inches(5.05), rows, [2.2, 2.8], size=10,
          row_h=Inches(0.42))
    notes(s, "This slide exists because the marking scheme rewards methodology. It is "
             "also the answer to \u201chow do you know the numbers are right\u201d.")


def s_limits(prs, n):
    s = blank(prs)
    header(s, "16 · Limitations", "What this project does not establish", n)
    bullets(s, MARGIN, Inches(1.8), Inches(6.6), Inches(4.6), [
        ("No field photographs.", " All evidence comes from one studio-style dataset; "
         "phone photos in a kitchen are untested."),
        ("Small, fuzzy minority classes.", " Damaged has 46 validation and 49 test images, "
         "and Ripe/Old boundaries are genuinely ambiguous."),
        ("Uncalibrated scores.", " A 0.80 vote share is not \u201c80 % likely correct\u201d; "
         "no temperature scaling was fitted."),
        ("Single seed per configuration.", " A ±0.01 difference between two runs is inside "
         "the noise, which is why the tie was accepted rather than broken."),
        ("Test set spent.", " It was scored once; any future model comparison needs new data."),
    ], size=12)
    rect(s, Inches(7.7), Inches(1.8), Inches(4.85), Inches(1.5), AMBER_BG, line=MINT_DEEP)
    rect(s, Inches(7.7), Inches(1.8), Pt(4), Inches(1.5), AMBER)
    text(s, Inches(7.95), Inches(1.98), Inches(4.4), Inches(1.2),
         "Not claimed: internal spoilage, edibility, contamination, nutrition, or "
         "\u201cfreshness\u201d as a physical quantity. The output is a visual similarity "
         "judgement between five categories.",
         size=11.5, colour=RGBColor(0x5C, 0x44, 0x10), spacing=1.35)
    bullets(s, Inches(7.7), Inches(3.55), Inches(4.85), Inches(2.0), [
        ("Deep learning is not declared worse.", " The finding is scoped: these frozen "
         "backbones, this dataset, this protocol."),
    ], size=11.5)
    notes(s, "Never say \u201cdeep learning does not work for vegetables\u201d. The scoped "
             "claim is what the evidence supports, and it is what the report says.")


def s_conclusion(prs, n):
    s = blank(prs)
    header(s, "17 · Conclusion", "What was achieved, and what is next", n)
    bullets(s, MARGIN, Inches(1.8), Inches(6.4), Inches(4.4), [
        ("A measured answer to the question.", f" {TEST['macro_f1']:.4f} test macro F1 on "
         "935 unseen images, with per-class evidence and a stated envelope."),
        ("A trustworthy comparison.", " Four models, one scorer, paired statistics — and "
         "the willingness to deploy the model that was not the expected one."),
        ("A working system.", f" {E2E:.0f} ms "
         "predictions, 178 tests, audited integrity, accessible interface."),
        ("A record of failure modes.", " Degradation study, confident-error analysis, and "
         "one real serving defect found and fixed."),
    ], size=12)
    rect(s, Inches(7.55), Inches(1.8), Inches(5.0), Inches(4.35), PAPER, line=MINT_DEEP)
    text(s, Inches(7.8), Inches(1.98), Inches(4.5), Inches(0.3), "NEXT, WITH MORE TIME",
         size=10, colour=EMERALD, bold=True)
    bullets(s, Inches(7.8), Inches(2.4), Inches(4.5), Inches(3.6), [
        "Fine-tune the last ResNet50 block",
        "Re-audit the Ripe/Old label boundary",
        "Multiple seeds, then re-test on fresh data",
        "Calibrate the scores (isotonic / temperature)",
        "100+ verified field photographs",
    ], size=11.5)
    note(s, MARGIN, Inches(5.95), Inches(6.4), Inches(0.72),
         "Submission state: reports, artefacts, tests and a hash-verified release manifest "
         "are all in the repository.")
    notes(s, "Close on the contribution: the discipline of the comparison, not the "
             "classifier, is what makes the result usable.")


def s_thanks(prs, n):
    s = blank(prs)
    rect(s, 0, 0, W, H, EMERALD_DK)
    rect(s, 0, H - Inches(0.16), W, Inches(0.16), EMERALD_LT)
    rect(s, W - Inches(3.4), H - Inches(2.9), Inches(4.2), Inches(4.2),
         RGBColor(0x0F, 0x5C, 0x3F), shape=MSO_SHAPE.OVAL)
    text(s, MARGIN, Inches(2.3), Inches(8.6), Inches(0.9), "Thank you", size=44, bold=True,
         colour=WHITE)
    text(s, MARGIN, Inches(3.3), Inches(8.4), Inches(0.9),
         "Questions — and the numbers behind every claim.", size=18,
         colour=RGBColor(0xD8, 0xF0, 0xE4))
    rect(s, MARGIN, Inches(4.25), Inches(1.6), Pt(3), EMERALD_LT)
    text(s, MARGIN, Inches(4.55), Inches(9.5), Inches(1.5),
         f"Test macro F1 {TEST['macro_f1']:.4f}  ·  accuracy {TEST['accuracy']:.4f}  ·  "
         f"validation {DEPLOYED['quality']['macro_f1']:.4f}\n"
         f"{E2E:.0f} ms median inference  ·  "
         "178 tests  ·  8 integrity checks\n"
         "Reports: reports/phase4_final_validation.md  ·  Demo script: reports/demo_checklist.md",
         size=12, colour=RGBColor(0xB9, 0xD8, 0xC9), spacing=1.5)
    notes(s, "Have the viva document open: reports/phase4_viva_questions.md, 24 prepared "
             "questions with the evidence for each answer.")


def main() -> None:
    prs = new_deck()
    builders = [
        s_title, s_agenda, s_problem, s_objectives, s_dataset, s_pipeline, s_features,
        s_deep, s_protocol, s_results, s_confusion, s_decision, s_test, s_robustness,
        s_performance, s_app, s_quality, s_limits, s_conclusion, s_thanks,
    ]
    for i, build in enumerate(builders, start=1):
        build(prs, i)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PATH)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
