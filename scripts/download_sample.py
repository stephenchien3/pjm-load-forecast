"""Download a PJM hourly load CSV for one zone.

This helper is for users who want the full multi-year series. The bundled
``data/sample.csv`` is sufficient to run the tests and the demo notebook.

Usage:
    python scripts/download_sample.py --zone PJME --out data/PJME_hourly.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

# Public mirrors of the PJM hourly series. PJM Data Miner 2 itself requires
# an API key; for reproducibility this script points at a long-mirrored
# public copy of the same per-zone CSVs.
_MIRROR_TEMPLATE = (
    "https://raw.githubusercontent.com/panambY/Hourly_Energy_Consumption/master/data/{zone}_hourly.csv"
)
_KNOWN_ZONES = {"PJME", "PJMW", "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ", "EKPC", "FE", "NI"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zone", required=True, help=f"one of {sorted(_KNOWN_ZONES)}")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args(argv)

    if args.zone not in _KNOWN_ZONES:
        print(f"unknown zone {args.zone!r}; known: {sorted(_KNOWN_ZONES)}", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = _MIRROR_TEMPLATE.format(zone=args.zone)
    print(f"downloading {url} -> {out}")
    urlretrieve(url, out)  # noqa: S310
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
