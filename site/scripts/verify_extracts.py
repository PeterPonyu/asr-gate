#!/usr/bin/env python3
"""Verify stamped site/_data JSON against extracts.sha256."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "_data"
MANIFEST = DATA / "extracts.sha256"
SKIP = {"extracts.sha256.json"}


def main() -> int:
    if not MANIFEST.exists():
        print("no extracts yet")
        return 0
    expected = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split()
        expected[name] = digest
    json_files = sorted(p for p in DATA.glob("*.json") if p.name not in SKIP)
    names = {p.name for p in json_files}
    if names != set(expected):
        print("extract set mismatch", sorted(names), sorted(expected))
        return 1
    failed = False
    for name, digest in sorted(expected.items()):
        got = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
        if got != digest:
            print("hash drift", name)
            failed = True
        else:
            print("ok", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
