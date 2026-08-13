#!/usr/bin/env python3
"""GATE-LOCAL checks after Eleventy build (prefix, routes, numbers, no PDF)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "_site"
PREFIX = "/asr-gate/"
ROUTES = [
    "index.html",
    "method/index.html",
    "results/index.html",
    "limitations/index.html",
    "reproduce/index.html",
    "cite/index.html",
]
NEEDLES = [
    "85.7",
    "96.2",
    "1.00",
    "1.53",
    "0.012",
    "0.051",
    "0/20",
    "accept/defer",
    "vacuous-at-target",
]


def main() -> int:
    failed = False
    for rel in ROUTES:
        path = ROOT / rel
        if not path.exists():
            print("missing route", rel)
            failed = True
    html = "\n".join(p.read_text(errors="replace") for p in ROOT.rglob("*.html"))
    if "/css/" in html and f"{PREFIX}css/" not in html:
        print("unprefixed /css/ link")
        failed = True
    if f"{PREFIX}css/portal.css" not in html:
        print("missing prefixed stylesheet")
        failed = True
    if 'href="/css/' in html or "href='/css/" in html:
        print("root-relative unprefixed css href")
        failed = True
    for needle in NEEDLES:
        if needle not in html:
            print("missing number/claim", needle)
            failed = True
    pdfs = list(ROOT.rglob("paper_*.pdf"))
    if pdfs:
        print("manuscript pdf in artifact", pdfs)
        failed = True
    if not failed:
        print("gate-local html checks ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
