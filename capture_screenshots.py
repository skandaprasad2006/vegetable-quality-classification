"""Capture evidence screenshots of the running Streamlit application.

Drives the locally installed Chrome through Playwright (no browser download) and
exercises the real upload → predict flow before capturing it.

Usage (with the app already running on :8501):

    python tools/capture_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = "http://localhost:8501"
OUT = Path(__file__).resolve().parents[1] / "reports" / "screenshots"

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)


def _chrome_path() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise SystemExit("no local Chrome or Edge installation found")


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=_chrome_path(), headless=True)
        # Streamlit scrolls an inner div, so ``full_page`` cannot exceed the
        # viewport; a tall viewport is how the whole page gets captured.
        page = browser.new_page(viewport={"width": 1360, "height": 2400},
                                device_scale_factor=1)
        page.on("console", lambda m: console_errors.append(m.text)
                if m.type == "error" else None)
        page.on("response", lambda r: console_errors.append(f"HTTP {r.status} {r.url}")
                if r.status >= 400 else None)

        # 1. Landing page
        page.goto(BASE, wait_until="networkidle")
        page.wait_for_selector("text=VegCheck", timeout=60000)
        # The hero renders before the cached model finishes loading; wait for the
        # upload section so the capture is not taken mid-spinner.
        page.wait_for_selector("text=Provide an image", timeout=90000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "01_home.png"), full_page=True)
        print("captured 01_home.png")

        # 2. Sample-image prediction, driven through the real controls
        page.get_by_text("Use a sample validation image").click()
        page.wait_for_timeout(2000)
        page.get_by_role("button", name="Predict quality").click()
        page.wait_for_selector("text=Predicted quality", timeout=90000)
        page.wait_for_timeout(2500)
        page.screenshot(path=str(OUT / "02_prediction.png"), full_page=True)
        print("captured 02_prediction.png")

        # Result close-up: chart plus history, scrolled into place.
        page.mouse.wheel(0, 1500)
        page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "02b_result_detail.png"))
        print("captured 02b_result_detail.png")

        # 3. Secondary pages
        for slug, name in (
            ("Dataset_Information", "03_dataset.png"),
            ("Model_Performance", "04_model_performance.png"),
            ("About_Project", "05_about.png"),
        ):
            page.goto(f"{BASE}/{slug}", wait_until="networkidle")
            page.wait_for_timeout(3500)
            page.screenshot(path=str(OUT / name), full_page=True)
            print(f"captured {name}")

        browser.close()

    if console_errors:
        print("\nconsole errors:")
        for err in console_errors[:10]:
            print("  -", err)
    else:
        print("\nno console errors")


if __name__ == "__main__":
    sys.exit(main())
