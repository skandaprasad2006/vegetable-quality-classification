"""Phase 4 Stage 5 — accessibility and usability audit of the running app.

Measures what can be measured (contrast ratios, headings, focusable elements,
image alternatives, horizontal overflow, mobile layout) and captures evidence
screenshots. It does **not** claim WCAG conformance: that requires manual testing
with assistive technology by a qualified auditor.

Run (with the app on :8501):  python tools/accessibility_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = "http://localhost:8501"
ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "reports" / "screenshots" / "phase4"
OUT = ROOT / "evaluation" / "phase4_accessibility"

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)

#: Samples text elements and computes WCAG 2.1 contrast against the effective
#: background, walking ancestors until an opaque colour is found.
CONTRAST_JS = r"""
() => {
  const parse = (c) => {
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(',').map(x => parseFloat(x));
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const lum = ({r, g, b}) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
  };
  const effBg = (el) => {
    let node = el;
    while (node) {
      const bg = parse(getComputedStyle(node).backgroundColor);
      if (bg && bg.a > 0.9) return bg;
      node = node.parentElement;
    }
    return {r: 255, g: 255, b: 255, a: 1};
  };
  const ratio = (a, b) => {
    const l1 = lum(a), l2 = lum(b);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  };

  const samples = [];
  const pick = (sel, label) => {
    const el = document.querySelector(sel);
    if (!el) return;
    const cs = getComputedStyle(el);
    const fg = parse(cs.color);
    const bg = effBg(el);
    if (!fg) return;
    samples.push({
      label,
      text: (el.textContent || '').trim().slice(0, 60),
      fontSizePx: parseFloat(cs.fontSize),
      fontWeight: cs.fontWeight,
      color: cs.color,
      background: `rgb(${bg.r}, ${bg.g}, ${bg.b})`,
      contrast: Math.round(ratio(fg, bg) * 100) / 100,
    });
  };

  pick('h1', 'page title');
  pick('h3', 'section heading');
  pick('[data-testid="stMarkdownContainer"] p', 'body text');
  pick('[data-testid="stCaptionContainer"]', 'caption');
  pick('[data-testid="stMetricLabel"]', 'metric label');
  pick('[data-testid="stMetricValue"]', 'metric value');
  pick('button', 'button');
  pick('[data-testid="stSidebar"] a', 'sidebar nav link');
  pick('[data-testid="stAlertContainer"]', 'info box');

  const imgs = [...document.querySelectorAll('img')].map(i => ({
    alt: i.getAttribute('alt') || '',
    hasAlt: i.hasAttribute('alt'),
    width: i.naturalWidth,
  }));

  const headings = [...document.querySelectorAll('h1,h2,h3,h4')]
    .map(h => ({tag: h.tagName, text: (h.textContent || '').trim().slice(0, 70)}));

  const focusable = [...document.querySelectorAll(
    'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
  )].filter(el => el.offsetParent !== null).length;

  return {
    samples,
    images: imgs,
    headings,
    focusableCount: focusable,
    overflowX: document.documentElement.scrollWidth > window.innerWidth + 2,
    docWidth: document.documentElement.scrollWidth,
    viewportWidth: window.innerWidth,
  };
}
"""


def _chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("no Chrome/Edge found")


def _grade(contrast: float, large: bool) -> str:
    # WCAG 2.1 AA: 4.5:1 normal text, 3:1 for large text (>=18.66px bold or 24px)
    threshold = 3.0 if large else 4.5
    return "pass_AA" if contrast >= threshold else "below_AA"


def main() -> None:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"viewport_runs": [], "pages": {}}

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chrome(), headless=True)

        # --- desktop pass over every page --------------------------------
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        desktop = {}
        pages = [
            ("", "01_home", "VegCheck"),
            ("/Dataset_Information", "03_dataset", "Dataset information"),
            ("/Model_Performance", "04_model_performance", "Model performance"),
            ("/About_Project", "05_about", "About this project"),
        ]
        for slug, shot, expect in pages:
            page.goto(BASE + slug, wait_until="networkidle")
            page.wait_for_selector(f"text={expect}", timeout=60000)
            page.wait_for_timeout(2500 if slug == "" else 1500)
            page.screenshot(path=str(SHOTS / f"{shot}.png"), full_page=True)
            data = page.evaluate(CONTRAST_JS)
            for s in data["samples"]:
                s["grade"] = _grade(s["contrast"], s["fontSizePx"] >= 24)
            desktop[shot] = {
                "url": page.url,
                "samples": data["samples"],
                "images_without_alt": sum(1 for i in data["images"] if not i["hasAlt"]),
                "images_total": len(data["images"]),
                "headings": data["headings"],
                "focusable_elements": data["focusableCount"],
                "horizontal_overflow": data["overflowX"],
            }
            print(f"desktop {shot}: {len(data['samples'])} text samples, "
                  f"{data['focusableCount']} focusable, overflow={data['overflowX']}")

        # --- keyboard navigation smoke test ------------------------------
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector("text=VegCheck", timeout=60000)
        seen = []
        for _ in range(12):
            page.keyboard.press("Tab")
            seen.append(page.evaluate(
                "() => { const a = document.activeElement;"
                "return a ? (a.innerText || a.getAttribute('aria-label') || a.tagName).trim().slice(0,40) : null; }"))
        report["keyboard"] = {
            "tab_sequence": seen,
            "reachable_distinct": len({s for s in seen if s}),
            "nav_reachable": any("Dataset" in (s or "") or "Model" in (s or "") for s in seen),
        }
        print(f"keyboard: {report['keyboard']['reachable_distinct']} distinct stops, "
              f"nav reachable={report['keyboard']['nav_reachable']}")

        # --- prediction flow screenshot (desktop) ------------------------
        page.goto(BASE + "?sample=0&predict=1", wait_until="networkidle")
        page.wait_for_selector("text=Predicted quality", timeout=90000)
        page.wait_for_timeout(2500)
        page.mouse.wheel(0, 1400)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(SHOTS / "02_prediction.png"))
        print("captured 02_prediction.png")

        # --- mobile viewport --------------------------------------------
        mobile = browser.new_page(viewport={"width": 390, "height": 844},
                                  device_scale_factor=2, is_mobile=True,
                                  has_touch=True)
        mobile.goto(BASE, wait_until="networkidle")
        mobile.wait_for_selector("text=VegCheck", timeout=60000)
        mobile.wait_for_timeout(2000)
        mobile.screenshot(path=str(SHOTS / "06_mobile_home.png"), full_page=True)
        mdata = mobile.evaluate(CONTRAST_JS)
        report["mobile"] = {
            "viewport": [390, 844],
            "horizontal_overflow": mdata["overflowX"],
            "doc_width": mdata["docWidth"],
            "viewport_width": mdata["viewportWidth"],
            "focusable_elements": mdata["focusableCount"],
            "min_contrast": min((s["contrast"] for s in mdata["samples"]), default=None),
        }
        print(f"mobile: overflow={mdata['overflowX']} docWidth={mdata['docWidth']}")

        browser.close()

    report["pages"] = desktop

    # --- aggregate -------------------------------------------------------
    all_samples = [s for page_data in desktop.values() for s in page_data["samples"]]
    below = [s for s in all_samples if s["grade"] == "below_AA"]
    report["summary"] = {
        "text_samples": len(all_samples),
        "below_AA": len(below),
        "below_AA_detail": below,
        "min_contrast": min((s["contrast"] for s in all_samples), default=None),
        "pages_with_h1": sum(1 for d in desktop.values()
                             if any(h["tag"] == "H1" for h in d["headings"])),
        "pages": len(desktop),
        "images_without_alt": sum(d["images_without_alt"] for d in desktop.values()),
        "images_total": sum(d["images_total"] for d in desktop.values()),
        "horizontal_overflow_desktop": any(d["horizontal_overflow"] for d in desktop.values()),
        "note": ("Automated checks only. No WCAG conformance is claimed."),
    }

    (OUT / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- summary ---")
    print(f"text samples: {report['summary']['text_samples']}, "
          f"below AA: {report['summary']['below_AA']}, "
          f"min contrast: {report['summary']['min_contrast']}")
    for s in below:
        print(f"  BELOW AA {s['contrast']:.2f}  {s['label']}: {s['text'][:40]!r}")
    print(f"images without alt: {report['summary']['images_without_alt']}/"
          f"{report['summary']['images_total']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
