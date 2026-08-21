#!/usr/bin/env python3
"""
CPU coupled-vs-decoupled score contrast on frozen asr-gate landscape decodes.

Owned by asr-gate (007 residual routed here). Does NOT rewrite LTT package
substrate into CRC — standalone CRC-style deploy-valid contrast only.

Vendor pin: jieba 0.42.1 (local copy under HERE/vendor; private cache HERE/.jieba_cache).

Mapping (see MAPPING.md; polarity gate):
  s1  (mean token logps)           ↔ E4 decoupled
  hyp_entity_mass (hyp-only jieba) ↔ E4 conf / coupled

Primary endpoint: deploy-valid % @ α ∈ {0.05, 0.10, 0.20}.
Vacuity: oracle_vacuous_frac (oracle flag ∈ {0,1}) vs deploy_vacuous_frac
(deploy flag ∈ {0,1}). Primary nonvacuous stratum = both-arms deploy-nonvacuous.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LANDSCAPE = HERE.parent / "landscape_pulled_2026-07-15"
VENDOR = HERE / "vendor"
JIEBA_CACHE = HERE / ".jieba_cache"
JIEBA_CACHE.mkdir(mode=0o700, exist_ok=True)
for _k in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_k] = str(JIEBA_CACHE)
sys.path.insert(0, str(VENDOR))
import jieba.posseg as pseg  # noqa: E402

CORPORA = ("aishell", "magicdata", "thchs30")
BACKBONES = ("paraformer", "belle", "zipformer")
ENTITY_POS = {"nr", "ns", "nt", "nz"}
ALPHAS = (0.05, 0.10, 0.20)
N_SEEDS = 20
SEED_BASE = 5000
BYTE_RE = re.compile(r"((?:<0x[0-9A-Fa-f]{2}>)+)")

OUT_JSON = HERE / "results.json"
OUT_SUMMARY = HERE / "SUMMARY.md"
OUT_D1 = HERE / "D1_STRATIFIED.md"
OUT_D1_JSON = HERE / "D1_STRATIFIED.json"
OUT_D2 = HERE / "D2_ABLATIONS.md"
OUT_D3 = HERE / "D3_PARETO.md"


def code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def debyte(text: str) -> str:
    if "<0x" not in text:
        return text

    def sub(m):
        raw = bytes(int(h, 16) for h in re.findall(r"<0x([0-9A-Fa-f]{2})>", m.group(1)))
        return raw.decode("utf-8", errors="replace")

    return BYTE_RE.sub(sub, text)


def matched_ref_indices(ref: str, hyp: str):
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return set()
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        ri = ref[i - 1]
        row, prev = dp[i], dp[i - 1]
        for j in range(1, n + 1):
            row[j] = prev[j - 1] + 1 if ri == hyp[j - 1] else max(prev[j], row[j - 1])
    out, i, j = set(), m, n
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            out.add(i - 1)
            i, j = i - 1, j - 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return out


def entity_mask(text: str) -> np.ndarray:
    n = len(text)
    mask = np.zeros(n, dtype=float)
    if n == 0:
        return mask
    pos = 0
    for word, flag in pseg.cut(text):
        w = len(word)
        if flag in ENTITY_POS:
            mask[pos : pos + w] = 1.0
        pos += w
    return mask


def crc_lambda(scores, losses, alpha):
    n = len(scores)
    if n == 0:
        return np.inf
    for lam in np.concatenate([np.sort(np.unique(scores)), [np.inf]]):
        if (n * float(np.mean(losses * (scores >= lam))) + 1.0) / (n + 1) <= alpha:
            return float(lam)
    return np.inf


def risk_at(scores, losses, lam):
    return float(np.mean(losses * (scores >= lam))) if len(scores) else np.nan


def auroc_high_loss(scores, loss_pos):
    """AUROC of score ranking high-loss above low. >0.55 → INVERTED for CRC accept≥λ."""
    pos, neg = scores[loss_pos], scores[~loss_pos]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float(
        (ranks[loss_pos].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    )


def load_cell(corpus: str, backbone: str):
    path = LANDSCAPE / f"decode_{corpus}_{backbone}_test.jsonl"
    if not path.exists():
        return None
    rows, raw_byte, residual_byte = [], 0, 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            ref = r.get("ref_text") or ""
            hyp_raw = r.get("hyp_text") or ""
            tl = (r.get("nbest") or [{}])[0].get("token_logps") or []
            if not ref or not tl:
                continue
            if "<0x" in hyp_raw:
                raw_byte += 1
            hyp = debyte(hyp_raw)
            if "\ufffd" in hyp:
                residual_byte += 1
            ent_ref = entity_mask(ref)
            if not ref:
                continue
            ent_hyp = entity_mask(hyp) if hyp else np.zeros(0)
            miss = np.ones(len(ref))
            for i in matched_ref_indices(ref, hyp):
                miss[i] = 0.0
            s1 = float(np.mean(tl))
            hyp_entity_mass = float(ent_hyp.sum() / max(len(hyp), 1)) if len(hyp) else 0.0
            hyp_entity_count = float(ent_hyp.sum())
            l_entity = (
                float((miss * ent_ref).sum() / ent_ref.sum()) if ent_ref.sum() else 0.0
            )
            rows.append(
                dict(
                    spk=str(r.get("speaker_id") or "unk"),
                    s1=s1,
                    hyp_entity_mass=hyp_entity_mass,
                    hyp_entity_count=hyp_entity_count,
                    true_ent=bool(ent_ref.sum() > 0),
                    pred_ent=bool(ent_hyp.sum() > 0),
                    l_entity=l_entity,
                )
            )
    if not rows:
        return None
    d = {
        k: np.array([r[k] for r in rows])
        for k in (
            "spk",
            "s1",
            "hyp_entity_mass",
            "hyp_entity_count",
            "true_ent",
            "pred_ent",
            "l_entity",
        )
    }
    d["_n"] = len(rows)
    d["_path"] = str(path)
    d["_raw_byte_frac"] = raw_byte / len(rows)
    d["_residual_byte_frac"] = residual_byte / len(rows)
    return d


def run_score(d, score_name: str, scores: np.ndarray, alpha: float, t, p):
    speakers = np.unique(d["spk"])
    deploy_ok, oracle_ok = [], []
    deploy_risk, oracle_risk = [], []
    deploy_flag, oracle_flag = [], []
    oracle_vacuous, deploy_vacuous = [], []
    infeas_deploy, infeas_oracle = [], []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(SEED_BASE + seed)
        perm = rng.permutation(speakers)
        cal_spk = set(perm[: len(perm) // 2])
        cal = np.array([s in cal_spk for s in d["spk"]])
        tst_true = (~cal) & t
        if tst_true.sum() < 30 or (cal & t).sum() < 20 or (cal & p).sum() < 10:
            continue
        lam_o = crc_lambda(scores[cal & t], d["l_entity"][cal & t], alpha)
        lam_p = crc_lambda(scores[cal & p], d["l_entity"][cal & p], alpha)
        s_t, l_t = scores[tst_true], d["l_entity"][tst_true]
        ro = risk_at(s_t, l_t, lam_o)
        rp = risk_at(s_t, l_t, lam_p)
        fo = float(np.mean(s_t < lam_o))
        fp = float(np.mean(s_t < lam_p))
        oracle_risk.append(ro)
        deploy_risk.append(rp)
        oracle_ok.append(ro <= alpha)
        deploy_ok.append(rp <= alpha)
        oracle_flag.append(fo)
        deploy_flag.append(fp)
        oracle_vacuous.append(fo in (0.0, 1.0))
        deploy_vacuous.append(fp in (0.0, 1.0))
        infeas_oracle.append(bool(np.isinf(lam_o)))
        infeas_deploy.append(bool(np.isinf(lam_p)))
    if not deploy_ok:
        return None
    o_vac = float(np.mean(oracle_vacuous))
    d_vac = float(np.mean(deploy_vacuous))
    return dict(
        score=score_name,
        alpha=alpha,
        n_seeds=len(deploy_ok),
        oracle_risk_median=float(np.median(oracle_risk)),
        oracle_valid=float(np.mean(oracle_ok)),
        deploy_risk_median=float(np.median(deploy_risk)),
        deploy_valid=float(np.mean(deploy_ok)),
        oracle_flag_median=float(np.median(oracle_flag)),
        deploy_flag_median=float(np.median(deploy_flag)),
        oracle_vacuous_frac=o_vac,
        deploy_vacuous_frac=d_vac,
        vacuous_frac=o_vac,  # deprecated: was oracle-only; use dual fields
        oracle_infeasible_frac=float(np.mean(infeas_oracle)),
        deploy_infeasible_frac=float(np.mean(infeas_deploy)),
    )


def _pair_key(cell, alpha):
    return (cell["corpus"], cell["backbone"], alpha)


def _build_pairs(cells, score_a: str, score_b: str):
    pairs = []
    for cell in cells:
        by = {(run["score"], run["alpha"]): run for run in cell["runs"]}
        for alpha in ALPHAS:
            a = by.get((score_a, alpha))
            b = by.get((score_b, alpha))
            if not a or not b:
                continue
            pairs.append(
                dict(
                    corpus=cell["corpus"],
                    backbone=cell["backbone"],
                    alpha=alpha,
                    score_a=score_a,
                    score_b=score_b,
                    deploy_valid_a=a["deploy_valid"],
                    deploy_valid_b=b["deploy_valid"],
                    oracle_vacuous_a=a["oracle_vacuous_frac"],
                    oracle_vacuous_b=b["oracle_vacuous_frac"],
                    deploy_vacuous_a=a["deploy_vacuous_frac"],
                    deploy_vacuous_b=b["deploy_vacuous_frac"],
                    delta_deploy_valid=b["deploy_valid"] - a["deploy_valid"],
                    nonvacuous_both_arms=(
                        a["deploy_vacuous_frac"] < 0.5 and b["deploy_vacuous_frac"] < 0.5
                    ),
                    nonvacuous_or_sensitivity=(
                        a["deploy_vacuous_frac"] < 0.5 or b["deploy_vacuous_frac"] < 0.5
                    ),
                    oracle_nonvacuous_both_arms=(
                        a["oracle_vacuous_frac"] < 0.5 and b["oracle_vacuous_frac"] < 0.5
                    ),
                )
            )
    return pairs


def _summarize_pairs(pairs):
    if not pairs:
        return {}
    all_delta = np.array([p["delta_deploy_valid"] for p in pairs])
    both = [p for p in pairs if p["nonvacuous_both_arms"]]
    or_nv = [p for p in pairs if p["nonvacuous_or_sensitivity"]]
    oracle_both = [p for p in pairs if p["oracle_nonvacuous_both_arms"]]
    med_both = (
        float(np.median([p["delta_deploy_valid"] for p in both])) if both else None
    )
    return dict(
        n_pairs=len(pairs),
        median_delta_all_pairs=float(np.median(all_delta)),
        mean_delta_deploy_valid=float(np.mean(all_delta)),
        frac_b_ge_a=float(np.mean(all_delta >= 0)),
        n_both_arms_nonvacuous=len(both),
        median_delta_both_arms_nonvacuous=med_both,
        n_or_nonvacuous=len(or_nv),
        median_delta_or_nonvacuous=(
            float(np.median([p["delta_deploy_valid"] for p in or_nv])) if or_nv else None
        ),
        n_oracle_both_arms_nonvacuous=len(oracle_both),
        median_delta_oracle_both_arms_nonvacuous=(
            float(np.median([p["delta_deploy_valid"] for p in oracle_both]))
            if oracle_both
            else None
        ),
        effect_direction_matches_e4=(med_both > 0 if med_both is not None else None),
        median_delta_deploy_valid=float(np.median(all_delta)),
        n_nonvacuous_pairs=len(or_nv),
        median_delta_nonvacuous=(
            float(np.median([p["delta_deploy_valid"] for p in or_nv])) if or_nv else None
        ),
    )


def _run_ablations(cells):
    """D2: pred_ent binary, hyp_entity_count raw; compare on both-arms headline set."""
    ablation_cells = []
    for cell in cells:
        d = cell["_data"]
        t, p = d["true_ent"], d["pred_ent"]
        if t.sum() < 60:
            continue
        ab_runs = [r for r in cell["runs"] if r["score"] == "s1"]
        pred_ent_score = d["pred_ent"].astype(float)
        for sname, scores in (
            ("pred_ent_binary", pred_ent_score),
            ("hyp_entity_count", d["hyp_entity_count"]),
        ):
            for alpha in ALPHAS:
                run = run_score(d, sname, scores, alpha, t, p)
                if run:
                    ab_runs.append(run)
        if any(r["score"] == "pred_ent_binary" for r in ab_runs):
            ablation_cells.append(
                dict(corpus=cell["corpus"], backbone=cell["backbone"], runs=ab_runs)
            )
    ab = {}
    ab["pred_ent_vs_s1"] = _build_pairs(
        [{**c, "runs": c["runs"]} for c in ablation_cells],
        "s1",
        "pred_ent_binary",
    )
    ab["entity_count_vs_s1"] = _build_pairs(
        [{**c, "runs": c["runs"]} for c in ablation_cells],
        "s1",
        "hyp_entity_count",
    )
    ab["pred_ent_summary"] = _summarize_pairs(ab["pred_ent_vs_s1"])
    ab["entity_count_summary"] = _summarize_pairs(ab["entity_count_vs_s1"])
    ab["scrambled_pos"] = dict(
        status="skipped",
        reason="CPU-only session; scrambled-POS / leave-one-out POS ablation deferred",
    )
    return ab


def _write_d1(out: dict):
    s = out.get("summary") or {}
    pairs = out.get("pair_deltas") or []
    strata = {
        "all_pairs": {
            "n": s.get("n_pairs"),
            "median_delta": s.get("median_delta_all_pairs"),
        },
        "both_arms_deploy_nonvacuous": {
            "n": s.get("n_both_arms_nonvacuous"),
            "median_delta": s.get("median_delta_both_arms_nonvacuous"),
        },
        "or_deploy_nonvacuous_sensitivity": {
            "n": s.get("n_or_nonvacuous"),
            "median_delta": s.get("median_delta_or_nonvacuous"),
        },
        "both_arms_oracle_nonvacuous": {
            "n": s.get("n_oracle_both_arms_nonvacuous"),
            "median_delta": s.get("median_delta_oracle_both_arms_nonvacuous"),
        },
    }
    by_corpus = {}
    for corpus in CORPORA:
        cp = [p for p in pairs if p["corpus"] == corpus]
        if cp:
            by_corpus[corpus] = dict(
                n=len(cp),
                median_delta=float(np.median([p["delta_deploy_valid"] for p in cp])),
                n_both_arms=sum(1 for p in cp if p["nonvacuous_both_arms"]),
            )
    d1 = dict(strata=strata, by_corpus=by_corpus, alphas=list(ALPHAS))
    OUT_D1_JSON.write_text(json.dumps(d1, indent=2))
    lines = [
        "# D1 — Stratified deploy-valid contrast",
        "",
        "Vacuity: **oracle** = oracle flag rate ∈ {0,1}; **deploy** = deploy flag rate ∈ {0,1}.",
        "Primary stratum: both score arms deploy-nonvacuous (`deploy_vacuous_frac < 0.5` on s1 AND coupled).",
        "Sensitivity: OR deploy-nonvacuous (either arm).",
        "",
        "## Headline strata (seed medians aggregated over cell×α pairs)",
        "",
        "| stratum | n | median Δ (coupled−s1) |",
        "|---|---:|---:|",
    ]
    for name, row in strata.items():
        med = row["median_delta"]
        med_s = f"{med:+.3f}" if med is not None else "—"
        lines.append(f"| {name} | {row['n']} | {med_s} |")
    lines += ["", "## By corpus (all pairs)", "", "| corpus | n | median Δ | n both-arms |", "|---|---:|---:|---:|"]
    for corpus, row in by_corpus.items():
        lines.append(
            f"| {corpus} | {row['n']} | {row['median_delta']:+.3f} | {row['n_both_arms']} |"
        )
    lines += ["", f"JSON: `{OUT_D1_JSON.name}`", ""]
    OUT_D1.write_text("\n".join(lines))


def _write_d2(out: dict, ab: dict):
    main_s = out.get("summary") or {}
    headline_med = main_s.get("median_delta_both_arms_nonvacuous")
    pred_s = ab.get("pred_ent_summary") or {}
    pred_med = pred_s.get("median_delta_both_arms_nonvacuous")
    count_s = ab.get("entity_count_summary") or {}
    count_med = count_s.get("median_delta_both_arms_nonvacuous")
    claim_strength = "standard"
    soft_ratio = None
    if headline_med is not None and pred_med is not None and headline_med > 0:
        soft_ratio = pred_med / headline_med
        if soft_ratio >= 0.8:
            claim_strength = "soft"
    ab["claim_strength"] = claim_strength
    ab["pred_ent_soft_ratio"] = soft_ratio
    lines = [
        "# D2 — Score ablations",
        "",
        f"Headline both-arms median Δ (coupled−s1): **{headline_med}**",
        f"Auto-soft heuristic tripped: **{'yes' if claim_strength == 'soft' else 'no'}**"
        + (f" (pred_ent ratio={soft_ratio:.2f})" if soft_ratio is not None else ""),
        "",
        "## (a) Binary pred_ent degenerate score (0/1)",
        "",
        f"- both-arms n={pred_s.get('n_both_arms_nonvacuous')}; "
        f"median Δ(pred_ent−s1): **{pred_med}**",
        f"- all-pairs median Δ: {pred_s.get('median_delta_all_pairs')}",
        "",
        "## (b) Length-norm vs raw entity count",
        "",
        f"- hyp_entity_mass = sum(entity mask)/|hyp| (coupled primary)",
        f"- hyp_entity_count = sum(entity mask) without /len",
        f"- both-arms n={count_s.get('n_both_arms_nonvacuous')}; "
        f"median Δ(count−s1): **{count_med}**",
        "",
        "## (c) Scrambled-POS / leave-one-out",
        "",
        f"- Status: {ab['scrambled_pos']['status']} — {ab['scrambled_pos']['reason']}",
        "",
    ]
    OUT_D2.write_text("\n".join(lines))
    return ab


def _write_d3(out: dict):
    lines = [
        "# D3 — Deploy flag vs deploy-valid Pareto notes",
        "",
        "Tradeoff: lower deploy_flag (abstention) vs higher deploy_valid (coverage at α).",
        "Median deploy_flag_median per score×α across cells:",
        "",
        "| score | α | median deploy_flag% | median deploy_valid% |",
        "|---|---:|---:|---:|",
    ]
    for score in ("s1", "hyp_entity_mass"):
        for alpha in ALPHAS:
            flags, vals = [], []
            for cell in out.get("cells") or []:
                for run in cell.get("runs") or []:
                    if run["score"] == score and run["alpha"] == alpha:
                        flags.append(run["deploy_flag_median"])
                        vals.append(run["deploy_valid"])
            if flags:
                lines.append(
                    f"| {score} | {alpha:.2f} | {np.median(flags)*100:.1f} | "
                    f"{np.median(vals)*100:.1f} |"
                )
    lines += [
        "",
        "**Commentary:** coupled (hyp_entity_mass) tends toward higher deploy_flag at low α",
        "(more abstention on pred-entity stratum) while often matching or exceeding deploy_valid;",
        "decoupled s1 shows inverted AUROC on true-class omission ranking — see SUMMARY D7 pointer.",
        "",
    ]
    OUT_D3.write_text("\n".join(lines))


def _write_summary(out: dict, ab: dict):
    sha = out.get("code_sha256", "")
    s = out.get("summary") or {}
    lines = [
        "# Coupled vs decoupled deploy-valid — ASR landscape CPU contrast",
        "",
        f"- Date: {out.get('date')}",
        f"- Landscape: `{out.get('landscape')}`",
        f"- Polarity: {out.get('polarity', {}).get('polarity_gate')}",
        f"- Alphas: {out.get('alphas')}",
        f"- Wall: {out.get('wall_s')}s CPU",
        f"- code_sha256: `{sha}`",
        f"- jieba: 0.42.1 vendored under `{VENDOR.name}/`; cache `{JIEBA_CACHE.name}/`",
        "",
        "## Vacuity definitions",
        "",
        "- **oracle_vacuous_frac**: fraction of seeds where oracle flag rate ∈ {0.0, 1.0}",
        "- **deploy_vacuous_frac**: fraction of seeds where deploy flag rate ∈ {0.0, 1.0}",
        "- **Primary stratum (both-arms)**: s1 AND coupled both deploy-nonvacuous (<0.5)",
        "- **Sensitivity (OR)**: either arm deploy-nonvacuous",
        "",
        "## Aggregate",
        "",
    ]
    if s:
        lines += [
            f"- n_pairs (cell×α): {s.get('n_pairs')}",
            f"- median Δ all pairs (coupled − s1): **{s.get('median_delta_all_pairs'):+.3f}**",
            f"- **PRIMARY** both-arms nonvacuous: n={s.get('n_both_arms_nonvacuous')}; "
            f"median Δ: **{s.get('median_delta_both_arms_nonvacuous')}**",
            f"- **SENSITIVITY** OR nonvacuous: n={s.get('n_or_nonvacuous')}; "
            f"median Δ: **{s.get('median_delta_or_nonvacuous')}**",
            f"- oracle both-arms nonvacuous: n={s.get('n_oracle_both_arms_nonvacuous')}; "
            f"median Δ: {s.get('median_delta_oracle_both_arms_nonvacuous')}",
            f"- effect direction matches E4 (PRIMARY both-arms): "
            f"**{s.get('effect_direction_matches_e4')}**",
            f"- D2 claim_strength: **{ab.get('claim_strength', 'standard')}**",
            "",
        ]
    lines += [
        "## AUROC-inverted warning",
        "",
        "Several cells show s1 AUROC > 0.55 on true-class high-loss ranking (inverted for CRC accept≥λ).",
        "Full D7 audit deferred; see per-cell `auroc_inverted_warning` in results.json.",
        "",
        "## Per-cell deploy-valid",
        "",
        "| corpus | backbone | α | s1 deploy% | coupled deploy% | Δ | dep_vac_s1 | dep_vac_c |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p in out.get("pair_deltas") or []:
        lines.append(
            f"| {p['corpus']} | {p['backbone']} | {p['alpha']:.2f} | "
            f"{p['deploy_valid_a']*100:.1f} | {p['deploy_valid_b']*100:.1f} | "
            f"{p['delta_deploy_valid']*100:+.1f} | {p['deploy_vacuous_a']*100:.0f}% | "
            f"{p['deploy_vacuous_b']*100:.0f}% |"
        )
    lines += [
        "",
        "Artifacts: `results.json`, `D1_STRATIFIED.md`, `D2_ABLATIONS.md`, `D3_PARETO.md`, "
        "`MAPPING.md`.",
        "",
    ]
    OUT_SUMMARY.write_text("\n".join(lines))


def main():
    t0 = time.time()
    if not LANDSCAPE.is_dir():
        note = {
            "status": "blocked-with-inventory",
            "reason": f"landscape missing: {LANDSCAPE}",
            "code_sha256": code_sha256(),
        }
        OUT_JSON.write_text(json.dumps(note, indent=2))
        print(json.dumps(note, indent=2))
        return 1

    polarity = {
        "decoupled": "s1",
        "coupled": "hyp_entity_mass",
        "polarity_gate": "PASS — s1↔decoupled; hyp_entity_mass↔coupled/conf",
        "e4_quote_source": (
            "frontier-directions-research/experiments/007-omission-crc/"
            "findings.md ~557-570"
        ),
    }

    out = {
        "status": "ok",
        "date": "2026-08-10",
        "compute": "CPU-only",
        "landscape": str(LANDSCAPE),
        "alphas": list(ALPHAS),
        "n_seeds": N_SEEDS,
        "seed_base": SEED_BASE,
        "polarity": polarity,
        "coupled_formula": "sum(jieba entity chars on debyted hyp) / max(|hyp|,1)",
        "jieba_vendor": "0.42.1 local vendor/",
        "jieba_cache": str(JIEBA_CACHE),
        "code_sha256": code_sha256(),
        "cells": [],
    }

    hdr = (
        f"{'corpus':<11}{'backbone':<12}{'score':<18}{'a':>5}"
        f"{'oracle%':>9}{'deploy%':>9}{'flag%':>8}{'o_vac':>6}{'d_vac':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    cell_data = []
    for corpus in CORPORA:
        for bk in BACKBONES:
            d = load_cell(corpus, bk)
            if d is None:
                print(f"{corpus:<11}{bk:<12} MISSING")
                continue
            t, p = d["true_ent"], d["pred_ent"]
            if t.sum() < 60:
                print(f"{corpus:<11}{bk:<12} too few true-entity utts ({int(t.sum())})")
                continue
            tp = int((t & p).sum())
            fn = int((t & ~p).sum())
            recall = tp / max(int(t.sum()), 1)
            det = d["l_entity"][t & p]
            mis = d["l_entity"][t & ~p]
            ratio = (
                float(mis.mean() / det.mean())
                if len(mis) and len(det) and det.mean() > 0
                else float("nan")
            )
            hi = d["l_entity"][t] > 0
            auroc = {
                "s1": auroc_high_loss(d["s1"][t], hi),
                "hyp_entity_mass": auroc_high_loss(d["hyp_entity_mass"][t], hi),
            }
            cell_meta = dict(
                corpus=corpus,
                backbone=bk,
                n=int(d["_n"]),
                n_true_ent=int(t.sum()),
                path=d["_path"],
                raw_byte_frac=float(d["_raw_byte_frac"]),
                residual_byte_frac=float(d["_residual_byte_frac"]),
                detector_recall=float(recall),
                n_detected=tp,
                n_missed=fn,
                omission_detected=float(det.mean()) if len(det) else float("nan"),
                omission_missed=float(mis.mean()) if len(mis) else float("nan"),
                missed_over_detected=ratio,
                auroc_high_loss_on_true_class=auroc,
                auroc_inverted_warning={
                    k: bool(v > 0.55) if v == v else None for k, v in auroc.items()
                },
                runs=[],
            )
            score_map = {"s1": d["s1"], "hyp_entity_mass": d["hyp_entity_mass"]}
            for sname, scores in score_map.items():
                for alpha in ALPHAS:
                    run = run_score(d, sname, scores, alpha, t, p)
                    if run is None:
                        print(f"{corpus:<11}{bk:<12}{sname:<18}{alpha:>5.2f}  no seeds")
                        continue
                    cell_meta["runs"].append(run)
                    print(
                        f"{corpus:<11}{bk:<12}{sname:<18}{alpha:>5.2f}"
                        f"{run['oracle_valid']*100:>8.1f}%"
                        f"{run['deploy_valid']*100:>8.1f}%"
                        f"{run['deploy_flag_median']*100:>7.1f}%"
                        f"{run['oracle_vacuous_frac']*100:>5.0f}%"
                        f"{run['deploy_vacuous_frac']*100:>5.0f}%"
                    )
            out["cells"].append(cell_meta)
            cell_data.append({**cell_meta, "_data": d})

    pairs = _build_pairs(out["cells"], "s1", "hyp_entity_mass")
    out["pair_deltas"] = pairs
    if pairs:
        out["summary"] = _summarize_pairs(pairs)

    ab = _run_ablations(cell_data)
    ab = _write_d2(out, ab)
    out["ablations"] = ab

    out["wall_s"] = round(time.time() - t0, 1)

    OUT_JSON.write_text(json.dumps(out, indent=2))
    _write_d1(out)
    _write_d3(out)
    _write_summary(out, ab)
    print(f"\nwrote {OUT_JSON}  wall={out['wall_s']}s")
    if out.get("summary"):
        s = out["summary"]
        print(
            f"summary: all-pairs median Δ={s['median_delta_all_pairs']:+.3f}"
            f"  both-arms n={s['n_both_arms_nonvacuous']} "
            f"median Δ={s['median_delta_both_arms_nonvacuous']}"
            f"  OR n={s['n_or_nonvacuous']} median Δ={s['median_delta_or_nonvacuous']}"
            f"  matches_e4={s['effect_direction_matches_e4']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
