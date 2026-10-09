#!/usr/bin/env python3
"""Download raw datasets.

Usage::

    python scripts/download_data.py --source sample      # generate built-in sample
    python scripts/download_data.py --source msd         # Million Song (auto download)
    python scripts/download_data.py --source msd --check # verify manual downloads

The Million Song taste-profile files are large (~1.5 GB total).  If the
automatic download fails, fetch them manually and place them in
``data/raw/msd/``:

* ``train_triplets.txt.gz`` or ``train_triplets.txt``
  (http://www.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip)
* ``unique_tracks.txt.gz``
  (http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt.gz)
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import bootstrap  # noqa: F401  (sys.path setup)

from ml.datasets.adapters.sample import SampleAdapter
from ml.utils.logging import get_logger, setup_logging
from ml.utils.paths import RAW_DATA_DIR, ensure_dirs

logger = get_logger("download_data")

MSD_DIR = RAW_DATA_DIR / "msd"
MSD_TRIPLETS_URL = "http://www.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip"
MSD_TRACKS_URL = "http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt.gz"


def _progress_hook(blocks: int, block_size: int, total: int) -> None:
    if total > 0 and blocks % 500 == 0:
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({100 * done / total:5.1f}%)", end="", flush=True)


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Already downloaded: %s", dest)
        return dest
    logger.info("Downloading %s -> %s", url, dest)
    urlretrieve(url, str(dest), reporthook=_progress_hook)
    print()
    return dest


def download_msd() -> None:
    MSD_DIR.mkdir(parents=True, exist_ok=True)

    tracks_gz = MSD_DIR / "unique_tracks.txt.gz"
    try:
        download_file(MSD_TRACKS_URL, tracks_gz)
    except Exception as exc:
        logger.error("Could not download unique_tracks (%s).", exc)
        logger.error("Download manually from %s and place it in %s", MSD_TRACKS_URL, tracks_gz)

    triplets_zip = MSD_DIR / "train_triplets.txt.zip"
    triplets_txt = MSD_DIR / "train_triplets.txt"
    if not triplets_txt.exists() and not (MSD_DIR / "train_triplets.txt.gz").exists():
        try:
            download_file(MSD_TRIPLETS_URL, triplets_zip)
            logger.info("Unzipping %s", triplets_zip)
            with zipfile.ZipFile(triplets_zip) as zf:
                zf.extractall(MSD_DIR)
        except Exception as exc:
            logger.error("Could not download triplets (%s).", exc)
            logger.error("Download manually from %s, unzip, and place train_triplets.txt in %s",
                         MSD_TRIPLETS_URL, MSD_DIR)


def check_msd() -> bool:
    ok = True
    for stem in ("unique_tracks", "train_triplets"):
        found = any((MSD_DIR / f"{stem}{ext}").exists() for ext in (".txt", ".txt.gz"))
        logger.info("%s: %s", stem, "FOUND" if found else f"MISSING (expected in {MSD_DIR})")
        ok = ok and found
    return ok


def generate_sample() -> None:
    adapter = SampleAdapter(regenerate=False)
    dataset = adapter.load()
    logger.info("Sample dataset ready: %s", dataset.summary())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="sample", choices=["sample", "msd"])
    parser.add_argument("--check", action="store_true", help="only verify manual downloads for --source msd")
    args = parser.parse_args()

    setup_logging()
    ensure_dirs()

    if args.source == "sample":
        generate_sample()
        return 0
    if args.check:
        return 0 if check_msd() else 1
    download_msd()
    return 0 if check_msd() else 1


if __name__ == "__main__":
    sys.exit(main())
