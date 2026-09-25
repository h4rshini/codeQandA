"""Record the README demo GIF from a running web app (uses a cached example: free, repeatable).

  pip install -r requirements-dev.txt
  CODEQA_REPO_DESCRIPTION="a stock watchlist app" .venv/bin/uvicorn web.app:app --port 8000
  .venv/bin/python scripts/record_demo.py            # writes docs/demo.gif
"""
import argparse
import io
import time
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

QUESTION = "How does the volume signal decide that a day's trading volume is unusual?"
OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.gif"


class Recorder:
    def __init__(self, page, width: int):
        self.page, self.width, self.frames = page, width, []  # (image, timestamp)

    def snap(self):
        img = Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB")
        img = img.resize((self.width, round(img.height * self.width / img.width)), Image.LANCZOS)
        self.frames.append((img, time.perf_counter()))

    def film(self, seconds: float, fps: float = 8):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            self.snap()
            time.sleep(max(0, 1 / fps - (time.perf_counter() - t0)))

    def save(self, path: Path, hold_last: float = 3.0):
        # Frame duration = real time until the next frame; merge identical frames to save space.
        out, durations = [], []
        for i, (img, t) in enumerate(self.frames):
            nxt = self.frames[i + 1][1] if i + 1 < len(self.frames) else t + hold_last
            ms = max(40, round((nxt - t) * 1000))
            if out and not ImageChops.difference(out[-1], img).getbbox():
                durations[-1] += ms
            else:
                out.append(img)
                durations.append(ms)
        palette = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in out]
        path.parent.mkdir(parents=True, exist_ok=True)
        palette[0].save(path, save_all=True, append_images=palette[1:], duration=durations,
                        loop=0, optimize=True, disposal=1)
        return len(out), path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--width", type=int, default=900, help="GIF width in pixels")
    ap.add_argument("--dark", action="store_true", help="record the dark theme")
    args = ap.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")  # uses installed Chrome; no browser download
        page = browser.new_page(viewport={"width": 1280, "height": 860},
                                color_scheme="dark" if args.dark else "light")
        rec = Recorder(page, args.width)

        page.goto(args.url)
        page.wait_for_selector(".bar")
        rec.film(2.2)                                         # skyline grows, highlighter sweeps

        page.click("#q")
        for ch in QUESTION:                                   # typing, filmed every few keys
            page.keyboard.type(ch)
            if ch == " ":
                rec.snap()
        rec.film(0.4)
        page.click("#ask-btn")

        # Keep the skyline and the step trail in view together while the answer replays.
        page.wait_for_selector("#run:not([hidden])")
        time.sleep(0.8)                                       # let the page's own scroll finish
        page.evaluate("scrollTo({top: document.querySelector('.skyline-wrap').offsetTop - 24})")
        start = time.perf_counter()
        while page.query_selector(".src .ln") is None and time.perf_counter() - start < 30:
            rec.film(0.25)
        rec.film(2.0)                                         # hold: cited lines light up in the skyline

        # Finish on the answer with its cited lines highlighted.
        page.evaluate("scrollTo({top: document.querySelector('#answer-h').getBoundingClientRect().top + scrollY - 24, behavior: 'smooth'})")
        rec.film(2.5)
        browser.close()

    n, size = rec.save(OUT)
    print(f"Wrote {OUT} ({n} frames, {size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
