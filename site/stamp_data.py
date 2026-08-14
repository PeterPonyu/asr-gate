#!/usr/bin/env python3
"""Stamp site/_data extracts from frozen manuscript records.

Reads origin/main frozen JSON only. Writes curated tables for the companion
site. Does not touch manuscript figure PDFs. Does not invent numbers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RES = ROOT / "manuscripts" / "results"
DATA = SITE / "_data"


def load(path: Path):
    with path.open() as fh:
        return json.load(fh)


def dump(name: str, obj) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    text = json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text)


def pct(x, digits=1):
    if x is None:
        return None
    return round(x * 100, digits)


def pct_str(x, digits=1, suffix="%"):
    if x is None:
        return "—"
    return f"{pct(x, digits):.{digits}f}{suffix}"


def main() -> None:
    num = load(RES / "numbers.json")
    audit_ci = load(RES / "audit_ci.json")
    band_file = load(ROOT / "alpha015_2026-07-13" / "results.json")
    man_cal = load(ROOT / "mandarin_calsweep_2026-07-13" / "results.json")
    eng_cal = load(ROOT / "english_calsweep_2026-07-13" / "results.json")
    bel_cal = load(ROOT / "calsize_power_2026-07-15" / "results.json")

    at = num["main_attainment"]
    fr = num["main_alpha_frontier"]
    holm = num["holm_realized"]
    ea = num["expansion_attainment"]
    exp_audit = num["expansion_audit"]
    rc = num["risk_coverage"]
    landscape = num["landscape"]
    bind = next(r for r in band_file["binding_band_sweep"] if abs(r["alpha"] - 0.019) < 1e-9)

    excess = [row["excess_aurc"] for row in holm["rows"]]

    dump(
        "meta.json",
        {
            "title": (
                "Certified Transcription Triage: a Distribution-Free "
                "Accept/Defer Gate with an Audited Confidence Signal for Mandarin ASR"
            ),
            "author": "Zeyu Fu",
            "orcid": "0009-0001-8329-0108",
            "orcid_url": "https://orcid.org/0009-0001-8329-0108",
            "year": 2026,
            "repo_url": "https://github.com/PeterPonyu/asr-gate",
            "pages_url": "https://peterponyu.github.io/asr-gate/",
            "zenodo_doi": "10.5281/zenodo.21392289",
            "zenodo_url": "https://doi.org/10.5281/zenodo.21392289",
            "license": "MIT",
        },
    )

    dump(
        "claims.json",
        {
            "c1": {
                "id": "C1",
                "name": "Certified selective triage",
                "text": (
                    "Learn-then-Test certifies accepted-set macro-CER ≤ α at "
                    "confidence 1−δ, or reports vacuous-at-target instead of "
                    "silently accepting nothing."
                ),
            },
            "c2": {
                "id": "C2",
                "name": "Debunk-or-confirm audit",
                "text": (
                    "Field-standard confidence must beat analytic random deferral; "
                    "all twelve roster cells show positive excess-AURC with "
                    "speaker-blocked CIs excluding zero."
                ),
            },
            "thesis": [
                "Production ASR rarely needs every utterance auto-transcribed; it needs the accepted set to stay under a character-error target and the rest deferred to a human.",
                "The object is an accept/defer gate on frozen decode artifacts, not a new recognizer and not a set-valued conformal hypothesis list.",
                "Claim C1: Learn-then-Test certifies accepted-set macro-CER ≤ α at confidence 1−δ, or reports vacuous-at-target instead of silently accepting nothing.",
                "On Aishell-1 with a Paraformer backbone at α = 2%, δ = 0.1: 0/20 correlated-reseed violations, 85.7–96.2% acceptance, accepted-set macro-CER 1.00–1.53% (full-set 1.98%).",
                "Claim C2: field-standard confidence must beat analytic random deferral; all twelve roster cells show positive excess-AURC (0.012–0.051) with speaker-blocked CIs excluding zero.",
                "The certificate is backbone-contingent: Whisper Mandarin is honestly vacuous at every tight target; zipformer exposed no usable posteriors; Belle certifies on Aishell-1 only at α = 5%.",
                "Coverage versus accepted-set CER is the signature plot; the three-backbone × three-corpus heatmap is the map — not a product dashboard.",
                "The recorded objects are the accept/defer gate, deferral, coverage, vacuity, and frozen artifacts.",
            ],
        },
    )

    op = fr["0.02"]
    dump(
        "frontier.json",
        {
            "delta": at["delta"],
            "n_reseeds": at["n_reseeds"],
            "n_test": at["n_test"],
            "n_violations_alpha02": at["n_violations"],
            "acceptance_min": at["accepted_fraction_min"],
            "acceptance_max": at["accepted_fraction_max"],
            "acceptance_mean": at["accepted_fraction_mean"],
            "accepted_cer_min": at["accepted_cer_min"],
            "accepted_cer_max": at["accepted_cer_max"],
            "accepted_cer_mean": at["accepted_cer_mean"],
            "full_set_macro_cer": exp_audit["aishell_paraformer_clean"]["macro_cer"],
            "audit_subset_macro_cer": rc["paraformer_clean"]["full_macro_cer"],
            "whisper_aishell_macro_cer": exp_audit["aishell_whisper_clean"]["macro_cer"],
            "whisper_thchs_macro_cer": exp_audit["thchs30_whisper_crosscorpus"]["macro_cer"],
            "operating_point": {
                "alpha": 0.02,
                "reseed": 0,
                "coverage": op["test_accepted_fraction"],
                "accepted_macro_cer": op["test_accepted_macro_cer"],
                "certified": op["certified"],
            },
            "alphas": [
                {
                    "alpha": fr[k]["alpha"],
                    "certified": fr[k]["certified"],
                    "coverage": fr[k]["test_accepted_fraction"],
                    "accepted_macro_cer": fr[k]["test_accepted_macro_cer"],
                }
                for k in ["0.01", "0.02", "0.03", "0.05"]
            ],
            "display": {
                "acceptance_range": (
                    f"{pct_str(at['accepted_fraction_min'])}–{pct_str(at['accepted_fraction_max'])}"
                ),
                "acceptance_mean": pct_str(at["accepted_fraction_mean"]),
                "accepted_cer_range": (
                    f"{pct_str(at['accepted_cer_min'], 2)}–{pct_str(at['accepted_cer_max'], 2)}"
                ),
                "full_set_macro_cer": pct_str(
                    exp_audit["aishell_paraformer_clean"]["macro_cer"], 2
                ),
                "audit_subset_macro_cer": pct_str(
                    rc["paraformer_clean"]["full_macro_cer"], 2
                ),
                "op_coverage": pct_str(op["test_accepted_fraction"]),
                "op_cer": pct_str(op["test_accepted_macro_cer"], 2),
                "violations": f"{at['n_violations']}/{at['n_reseeds']}",
                "whisper_aishell": pct_str(
                    exp_audit["aishell_whisper_clean"]["macro_cer"], 1
                ),
                "whisper_thchs": pct_str(
                    exp_audit["thchs30_whisper_crosscorpus"]["macro_cer"], 2
                ),
            },
        },
    )

    rc_keep = {}
    for key in [
        "paraformer_clean",
        "whisper_clean",
        "whisper_thchs30",
    ]:
        d = rc[key]
        rc_keep[key] = {
            "n": d["n"],
            "coverage": d["coverage"],
            "risk": d["risk"],
            "oracle_risk": d["oracle_risk"],
            "random_line": d["random_line"],
            "full_macro_cer": d["full_macro_cer"],
        }
    dump("rc.json", rc_keep)

    cond_label = {
        "aishell_clean": "Aishell-1 clean",
        "musan5db": "Aishell-1 + ESC-50 5 dB",
        "musan15db": "Aishell-1 + ESC-50 15 dB",
        "musan25db": "Aishell-1 + ESC-50 25 dB",
        "thchs30_crosscorpus": "THCHS-30 (cross-corpus)",
    }
    score_label = {
        "s1": "log-posterior",
        "s2": "weak-link",
    }
    bb_label = {"paraformer": "Paraformer", "whisper": "Whisper"}

    holm_rows = []
    for row in holm["rows"]:
        key = f"{row['backbone']}:{row['condition']}:{row['score']}"
        ci = audit_ci["cells"][key]
        holm_rows.append(
            {
                "backbone": row["backbone"],
                "backbone_label": bb_label[row["backbone"]],
                "condition": row["condition"],
                "condition_label": cond_label[row["condition"]],
                "score": row["score"],
                "score_label": score_label[row["score"]],
                "excess_aurc": row["excess_aurc"],
                "p_holm": row["p_holm_global"],
                "reject_holm": row["reject_holm_global"],
                "ci_lo": ci["ci_lo"],
                "ci_hi": ci["ci_hi"],
                "ci_excludes_zero": ci["ci_excludes_zero"],
                "n": row["n"],
                "hover": (
                    f"{bb_label[row['backbone']]} / {cond_label[row['condition']]} / "
                    f"{score_label[row['score']]}: excess-AURC {row['excess_aurc']:.3f}; "
                    f"speaker-blocked 95% CI [{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]"
                ),
            }
        )

    dump(
        "holm.json",
        {
            "m": holm["m"],
            "family_alpha": holm["alpha"],
            "excess_min": min(excess),
            "excess_max": max(excess),
            "all_ci_exclude_zero": audit_ci["all_ci_exclude_zero"],
            "p_holm_floor": 0.006,
            "p_holm_note": (
                "p_Holm = 0.006 in every cell (permutation floor); "
                "magnitudes and CIs are the informative quantity."
            ),
            "display": {
                "excess_range": (
                    f"{min(excess):.3f}–{max(excess):.3f}"
                )
            },
            "rows": holm_rows,
        },
    )

    noise_rows = []
    noise_rows.append(
        {
            "label": "clean",
            "key": "clean",
            "accepted_cer_mean": at["accepted_cer_mean"],
            "accepted_cer_min": at["accepted_cer_min"],
            "accepted_cer_max": at["accepted_cer_max"],
            "coverage_mean": at["accepted_fraction_mean"],
            "violations": at["n_violations"],
            "n_reseeds": at["n_reseeds"],
            "vacuous": False,
        }
    )
    for label, key in [
        ("25 dB", "musan25db"),
        ("15 dB", "musan15db"),
        ("5 dB", "musan5db"),
    ]:
        v = ea[key]
        noise_rows.append(
            {
                "label": label,
                "key": key,
                "accepted_cer_mean": v["acc_set_macro_cer_mean"],
                "accepted_cer_min": v["acc_set_macro_cer_min"],
                "accepted_cer_max": v["acc_set_macro_cer_max"],
                "coverage_mean": v["acc_fraction_mean"],
                "violations": v["violations"],
                "n_reseeds": v["n_reseeds"],
                "vacuous": v["vacuous"],
            }
        )

    dump(
        "noise.json",
        {
            "axis_title": "ESC-50 additive",
            "alpha": 0.02,
            "rows": noise_rows,
            "display": {
                "snr5_violations": f"{ea['musan5db']['violations']}/{ea['musan5db']['n_reseeds']}",
                "snr5_coverage": pct_str(ea["musan5db"]["acc_fraction_mean"]),
                "snr25_violations": f"{ea['musan25db']['violations']}/{ea['musan25db']['n_reseeds']}",
                "snr15_violations": f"{ea['musan15db']['violations']}/{ea['musan15db']['n_reseeds']}",
            },
        },
    )

    att = landscape["attainment"]
    aud = landscape["audit"]
    grid = [0.015, 0.02, 0.03, 0.05, 0.1]
    bbs = [
        ("paraformer", "Paraformer"),
        ("belle", "Belle"),
        ("zipformer", "zipformer"),
    ]
    corps = [
        ("aishell", "Aishell-1"),
        ("thchs30", "THCHS-30"),
        ("magicdata", "MagicData"),
    ]
    cells = []
    for bb, bb_lab in bbs:
        for co, co_lab in corps:
            tight = None
            cell = None
            for a in grid:
                c = att[f"{bb}_{co}_a{a}"]
                if (not c["vacuous"]) and c["violations"] == 0:
                    tight = a
                    cell = c
                    break
            override = bb == "paraformer" and co == "aishell"
            if override:
                tight = 0.019
                acc = bind["mean_acceptance"]
                acc_cer = bind["mean_accepted_macro_cer"]
                vacuous = False
            elif tight is None:
                acc = None
                acc_cer = None
                vacuous = True
            else:
                acc = cell["acc_fraction_mean"]
                acc_cer = cell["acc_set_macro_cer_mean"]
                vacuous = False
            full = aud.get(f"{bb}_{co}", {})
            hover_bits = [
                f"{bb_lab} / {co_lab}",
                f"full-set macro-CER {pct_str(full.get('macro_cer'), 2)}",
            ]
            if vacuous:
                hover_bits.append("vacuous-at-target on the frozen α-grid (no usable posteriors)" if bb == "zipformer" else "vacuous-at-target on the frozen α-grid")
            else:
                hover_bits.append(f"tightest α {pct_str(tight, 1)}")
                hover_bits.append(f"mean acceptance {pct_str(acc)}")
                hover_bits.append(f"mean accepted-set macro-CER {pct_str(acc_cer, 2)}")
            cells.append(
                {
                    "backbone": bb,
                    "backbone_label": bb_lab,
                    "corpus": co,
                    "corpus_label": co_lab,
                    "vacuous": vacuous,
                    "tightest_alpha": tight,
                    "mean_acceptance": acc,
                    "mean_accepted_cer": acc_cer,
                    "full_set_macro_cer": full.get("macro_cer"),
                    "binding_override": override,
                    "hover": "; ".join(hover_bits),
                }
            )

    dump(
        "landscape.json",
        {
            "binding_alpha": 0.019,
            "binding_acceptance": bind["mean_acceptance"],
            "binding_accepted_cer": bind["mean_accepted_macro_cer"],
            "binding_certified": bind["certified"],
            "display": {
                "binding_alpha": "1.9%",
                "binding_acceptance": pct_str(bind["mean_acceptance"]),
                "binding_accepted_cer": pct_str(bind["mean_accepted_macro_cer"], 2),
            },
            "cells": cells,
        },
    )

    bel_head = bel_cal["headline_min_n_cal_for_cert_frac_90pct"]
    belle_cells = []
    for cell in bel_cal["cells"]:
        belle_cells.append(
            {
                "alpha": cell["alpha"],
                "n_cal": cell["n_cal"],
                "cert_fraction": cell["cert_fraction"],
            }
        )

    dump(
        "calsize.json",
        {
            "paraformer_aishell_crossover": man_cal["crossover_seed0"],
            "belle_pool_size": bel_cal["cal_pool_size"],
            "belle_min_n_cal_90pct": bel_head,
            "belle_power_cells": belle_cells,
            "english_wav2vec2_large_crossover": eng_cal["crossover"]["wav2vec2_large"],
            "english_note": (
                "English wav2vec 2.0 large / LibriSpeech is an English "
                "calibration-budget series, not a Mandarin landscape cell."
            ),
        },
    )

    files = sorted(
        p for p in DATA.glob("*.json") if p.name != "extracts.sha256.json"
    )
    hashes = {}
    for p in files:
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump("extracts.sha256.json", hashes)
    lines = [f"{digest}  {name}" for name, digest in sorted(hashes.items())]
    (DATA / "extracts.sha256").write_text("\n".join(lines) + "\n")
    print("stamped", ", ".join(p.name for p in files))


if __name__ == "__main__":
    main()
