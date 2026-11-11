"""Download the QuALITY v1.0.1 dataset (HTML-stripped splits) into data/quality.

Usage:
    python scripts/download_quality.py                 # download from GitHub
    python scripts/download_quality.py --from-local quality_repo   # copy from a clone of nyu-mll/quality
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"
SPLITS = ("train", "dev", "test")
NAME = "QuALITY.v1.0.1.htmlstripped.{split}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/quality"))
    ap.add_argument("--from-local", type=Path, help="Path to a local clone of nyu-mll/quality")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.from_local:
        src = args.from_local / "data" / "v1.0.1"
        for split in SPLITS:
            shutil.copy2(src / NAME.format(split=split), args.out)
    else:
        print(f"Downloading {URL} ...")
        with urllib.request.urlopen(URL, timeout=120) as resp:
            data = resp.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for split in SPLITS:
                member = next(n for n in zf.namelist() if n.endswith(NAME.format(split=split)))
                (args.out / NAME.format(split=split)).write_bytes(zf.read(member))
    for split in SPLITS:
        path = args.out / NAME.format(split=split)
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
