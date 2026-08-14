#!/usr/bin/env python3
"""Scan built HTML/SVG for theme-mix and engineering leaks."""
from __future__ import annotations

import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else SITE / "_site"
FORBIDDEN_FILE = SITE / "FORBIDDEN.txt"

ALLOW = (
    "github.com/PeterPonyu/asr-gate",
    "peterponyu.github.io/asr-gate",
)


def allowed(text: str) -> str:
    out = text
    for needle in ALLOW:
        out = out.replace(needle, "")
    return out


def main() -> int:
    patterns = [
        line.strip()
        for line in FORBIDDEN_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    files = list(ROOT.rglob("*.html")) + list(ROOT.rglob("*.svg"))
    if not files:
        print("no html/svg under", ROOT)
        return 1
    hits = []
    for path in files:
        blob = allowed(path.read_text(errors="replace"))
        for pat in patterns:
            if pat in blob:
                hits.append(f"{path.relative_to(ROOT)}: {pat}")
        if re.search(r"\\cite|\\ref|\\cref", blob):
            hits.append(f"{path.relative_to(ROOT)}: TeX cite/ref token")
    if hits:
        print("LEAK")
        print("\n".join(hits))
        return 1
    print(f"leak scan clean ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
