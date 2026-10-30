#!/usr/bin/env python3
"""Download public trail-cam sample MP4s into demo/samples/."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"

# Pinned direct MP4 URLs (eastern wildlife / trail-cam style). Verified at fetch time.
SAMPLE_URLS: list[tuple[str, str]] = [
    (
        "raccoon_nywolf.mp4",
        "https://nywolf.org/wp-content/uploads/2021/06/Trail_cam_video_2_raccoon_360p.mp4",
    ),
    (
        "deer_archive.mp4",
        "https://archive.org/download/red-mulberry-deer-.-5-d.-d.-teoli-jr.-a.-c./red%20mulberry%20deer%20.5%20D.D.%20Teoli%20Jr.%20A.C..mp4",
    ),
]


def download(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Prefer curl when available — more reliable on macOS Python SSL setups.
    import shutil
    import subprocess

    curl = shutil.which("curl")
    if curl:
        subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "-o", str(dest), url],
            check=True,
            timeout=timeout,
        )
        return

    context = None
    try:
        import certifi
        import ssl

        context = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        context = None

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TrailCamDemo/0.1 (educational; +https://github.com/tmestery/TrailCam)"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp, dest.open(
        "wb"
    ) as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download trail-cam demo sample videos.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for filename, url in SAMPLE_URLS:
        dest = SAMPLES_DIR / filename
        if dest.exists() and not args.force:
            print(f"skip (exists): {dest}")
            ok += 1
            continue
        print(f"downloading {filename} ...")
        try:
            download(url, dest)
        except Exception as exc:
            print(f"FAILED {filename}: {exc}", file=sys.stderr)
            print(
                "You can manually drop any .mp4 into demo/samples/ and run "
                "python demo/live_demo.py --source demo/samples/yourfile.mp4",
                file=sys.stderr,
            )
            if dest.exists():
                dest.unlink()
            continue
        size_mb = dest.stat().st_size / (1024 * 1024)
        if size_mb < 0.05:
            print(f"FAILED {filename}: file too small ({size_mb:.2f} MB); removing", file=sys.stderr)
            dest.unlink(missing_ok=True)
            continue
        print(f"wrote {dest} ({size_mb:.1f} MB)")
        ok += 1

    if ok == 0:
        raise SystemExit(
            "No samples downloaded. Drop MP4 files into demo/samples/ manually."
        )
    print(f"Done. {ok} sample(s) available in {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
