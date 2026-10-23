"""Download YOLO11 and YOLO26 nano/x segmentation weights from Ultralytics.

Usage: python scripts/download_seg_models.py [--out-dir models/base] [--force]
Weights are saved under v11/ and v26/ within the output directory.
Requires only the Python standard library.
"""

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def download_models(out_dir, force=False):
    failures = []
    for version in ("11", "26"):
        directory = out_dir / f"v{version}"
        directory.mkdir(parents=True, exist_ok=True)
        for size in ("n", "x"):
            name = f"yolo{version}{size}-seg.pt"
            destination = directory / name
            if destination.is_file() and destination.stat().st_size > 0 and not force:
                print(f"Already exists: {destination}", flush=True)
                continue

            url = f"https://huggingface.co/Ultralytics/YOLO{version}/resolve/main/{name}"
            partial = destination.with_suffix(".pt.part")
            print(f"Downloading {name} -> {destination}", flush=True)
            try:
                with urlopen(url, timeout=120) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                    expected = response.headers.get("Content-Length")
                actual = partial.stat().st_size
                if not actual or (expected is not None and actual != int(expected)):
                    raise OSError(f"Incomplete download: received {actual} bytes (expected {expected})")
                partial.replace(destination)
                print(f"Saved {destination} ({actual / 1024**2:.1f} MiB)", flush=True)
            except (OSError, URLError, ValueError) as exc:
                failures.append(name)
                print(f"Failed to download {name}: {exc}", file=sys.stderr)
            finally:
                partial.unlink(missing_ok=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models/base")
    parser.add_argument("--force", action="store_true", help="Download again even if weights exist")
    args = parser.parse_args()
    failures = download_models(args.out_dir, args.force)
    if failures:
        parser.exit(1, f"Downloads failed: {', '.join(failures)}\n")


if __name__ == "__main__":
    main()
