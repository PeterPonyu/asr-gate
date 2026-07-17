"""Command-line entry point: ``asr-gate {ingest,score,calibrate,apply,audit,report}``.

Every subcommand emits machine-readable JSON (``--out``, and to stdout with
``--json``). ``report`` is the exception: a compact JSON+Markdown summary
per the design's MVP scope (§2.1: "no HTML needed for MVP").

One documented deviation from the design doc's literal §2.1 CLI listing:
``calibrate`` gains a ``--tune`` argument (optional). The design doc's own
§2.3 mandates that s5/G2 be fit on a tune split and calibrated/certified on
a DIFFERENT (speaker-disjoint) cal split (critical correctness requirement
#2, enforced in ``gate.py``); the literal §2.1 listing shows only
``--instances cal.parquet`` with no second input, which would make that
separation impossible to express at the CLI layer. ``--tune`` is optional:
omitting it auto-splits ``--instances`` by speaker internally (see
``gate.calibrate_gate``), so the documented one-file invocation still
works; passing it explicitly is how the real two-cohort pilot (dev's cal
vs. tune 20-speaker carves) is run (see ``orchestration/run_pilot.sh``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from asr_gate import audit as _audit
from asr_gate import cer as _cer
from asr_gate import gate as _gate
from asr_gate import io as _io
from asr_gate import scores as _scores


def _emit(result: Dict[str, Any], out: Optional[str], as_json: bool, summary: str) -> None:
    payload = _io.to_jsonable(result)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    if as_json or not out:
        json.dump(payload, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        print(summary)


def _load_instances(path: str) -> List[Dict[str, Any]]:
    return _io.load_utterances(path)


def _score_and_cer(
    instances: List[Dict[str, Any]], numeral_policy: str
) -> List[Dict[str, Any]]:
    scored = _scores.score_table(instances)
    return _cer.compute_cer_batch(scored, numeral_policy=numeral_policy)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    utterances = _io.ingest(args.hyps, args.format, refs_path=args.refs)
    _io.write_jsonl(args.out, utterances)
    n_with_ref = sum(1 for u in utterances if u.get("ref_text") is not None)
    print(f"ingest: format={args.format} n={len(utterances)} n_with_ref={n_with_ref} -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def _cmd_score(args: argparse.Namespace) -> int:
    instances = _load_instances(args.instances)
    scored = _score_and_cer(instances, args.numeral_policy)
    keep = args.scores.split(",") if args.scores else list(_scores.PRIMARY_SCORES[:4])
    n_cer = sum(1 for u in scored if "cer" in u)
    _io.write_jsonl(args.out, scored)
    print(
        f"score: n={len(scored)} scores={keep} n_cer_computed={n_cer} "
        f"normalizer={_cer.NORMALIZER_VERSION} -> {args.out}"
    )
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------


def _cmd_calibrate(args: argparse.Namespace) -> int:
    cal_instances = _load_instances(args.instances)
    cal_instances = _score_and_cer(cal_instances, args.numeral_policy)

    tune_instances = None
    input_paths = [args.instances]
    if args.tune:
        tune_instances = _load_instances(args.tune)
        tune_instances = _score_and_cer(tune_instances, args.numeral_policy)
        input_paths.append(args.tune)

    strata = args.strata.split(",") if args.strata else None
    gate = _gate.calibrate_gate(
        cal_instances,
        tune_instances=tune_instances,
        alpha=args.alpha,
        delta=args.delta,
        g1_score=args.g1_score,
        g2_score=args.g2_score,
        guarantee=args.guarantee,
        strata=strata,
        fit_frac=args.fit_frac,
        min_stratum_n=args.min_stratum_n,
        numeral_policy=args.numeral_policy,
        n_grid=args.n_grid,
        min_accept_frac=args.min_accept_frac,
        ltt_procedure=args.ltt_procedure,
        ltt_p_value=args.ltt_pvalue,
        seed=args.seed,
        input_paths=input_paths,
    )
    summary = (
        f"calibrate: guarantee={gate['guarantee']} alpha={gate['alpha']} delta={gate['delta']} "
        f"lambda_star={gate['g1']['lambda_star']} certified={gate['g1']['certified']} "
        f"accepted_fraction={gate['g1']['accepted_fraction']:.3f} "
        f"n_fit={gate['n_fit']} n_cal={gate['n_cal']}"
    )
    _emit(gate, args.out, args.json, summary)
    return 0


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def _cmd_apply(args: argparse.Namespace) -> int:
    with open(args.gate, "r", encoding="utf-8") as f:
        gate = json.load(f)
    instances = _load_instances(args.instances)
    instances = _scores.score_table(instances)
    result = _gate.apply_gate(gate, instances)
    summary = (
        f"apply: n={result['n']} accept={result['n_accept']} defer={result['n_defer']} "
        f"ood_refuse={result['n_ood_refuse']}"
    )
    if result["domain_fingerprint_check"] and result["domain_fingerprint_check"]["warn"]:
        summary += (
            f" -- WARNING: domain shift ks_distance="
            f"{result['domain_fingerprint_check']['ks_distance']:.3f}"
        )
    _emit(result, args.out, args.json, summary)
    return 0


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


def _cmd_audit(args: argparse.Namespace) -> int:
    instances = _load_instances(args.instances)
    instances = _score_and_cer(instances, args.numeral_policy)
    score_names = args.scores.split(",") if args.scores else None
    result = _audit.run_audit(
        instances,
        score_names=score_names,
        backbone_field=args.backbone_field,
        n_perm=args.n_perm,
        alpha=args.alpha,
        seed=args.seed,
    )
    lines = [
        f"audit: n={result['n']} holm_family_size={result['holm_family_size']} "
        f"macro_cer={result['macro_cer']:.4f} micro_cer={result['micro_cer']['point']:.4f} "
        f"[{result['micro_cer']['ci'][0]:.4f}, {result['micro_cer']['ci'][1]:.4f}]"
    ]
    for r in result["results"]:
        lines.append(
            f"  {r['score']}/{r['backbone']}: excess_aurc={r['excess_aurc']:.4f} "
            f"p={r['p_value']:.4f} p_holm={r['p_holm']:.4f} reject_holm={r['reject_holm']}"
        )
    _emit(result, args.out, args.json, "\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# report (compact JSON + Markdown; no HTML in the MVP)
# ---------------------------------------------------------------------------


def _render_report_markdown(
    audit_result: Optional[Dict[str, Any]], gate_result: Optional[Dict[str, Any]]
) -> str:
    lines = ["# asr-gate report", ""]

    if gate_result is not None:
        g1 = gate_result["g1"]
        lines += [
            "## Certificate (G1, LTT)",
            "",
            f"- guarantee: `{gate_result['guarantee']}`",
            f"- alpha (target CER): {gate_result['alpha']}",
            f"- delta (failure prob): {gate_result['delta']}",
            f"- certified: **{g1['certified']}**",
            f"- lambda*: {g1['lambda_star']}",
            f"- accepted fraction: {g1['accepted_fraction']:.4f}",
            f"- n_cal: {gate_result['n_cal']}, n_fit(tune): {gate_result['n_fit']}",
            f"- normalizer: `{gate_result['normalizer_version']}`",
            "",
            "### Mondrian strata",
            "",
            "| stratum | n_cal | defer_always | G2 threshold |",
            "|---|---|---|---|",
        ]
        for k, n in gate_result["strata"]["counts"].items():
            defer = gate_result["strata"]["defer_always"].get(k)
            thr = gate_result["strata"]["thresholds"].get(k)
            lines.append(f"| {k} | {n} | {defer} | {thr} |")
        lines.append("")

    if audit_result is not None:
        lines += [
            "## Audit (excess-AURC, Holm m={})".format(audit_result["holm_family_size"]),
            "",
            f"- macro-CER: {audit_result['macro_cer']:.4f}",
            f"- micro-CER: {audit_result['micro_cer']['point']:.4f} "
            f"CI=[{audit_result['micro_cer']['ci'][0]:.4f}, "
            f"{audit_result['micro_cer']['ci'][1]:.4f}] "
            f"(speaker-blocked, n_blocks={audit_result['micro_cer']['n_blocks']})",
            f"- clip count: {audit_result['clip_count']}",
            "",
            "| score | backbone | n | excess-AURC | p | p (Holm) | reject (Holm) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in audit_result["results"]:
            lines.append(
                f"| {r['score']} | {r['backbone']} | {r['n']} | {r['excess_aurc']:.4f} | "
                f"{r['p_value']:.4f} | {r['p_holm']:.4f} | {r['reject_holm']} |"
            )
        lines.append("")

    return "\n".join(lines)


def _cmd_report(args: argparse.Namespace) -> int:
    audit_result = None
    gate_result = None
    if args.audit:
        with open(args.audit, "r", encoding="utf-8") as f:
            audit_result = json.load(f)
    if args.gate:
        with open(args.gate, "r", encoding="utf-8") as f:
            gate_result = json.load(f)
    if audit_result is None and gate_result is None:
        print("error: report needs --audit and/or --gate", file=sys.stderr)
        return 1

    summary_json = {"audit": audit_result, "gate": gate_result}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
    else:
        markdown = _render_report_markdown(audit_result, gate_result)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(markdown)
    print(f"wrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="asr-gate", description="Certified conformal ASR triage")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="adapt a raw decode dump into the canonical schema")
    p_ingest.add_argument("--hyps", required=True, help="raw decode-output JSONL")
    p_ingest.add_argument(
        "--format", required=True, choices=sorted(_io.FORMAT_ADAPTERS), help="decode format"
    )
    p_ingest.add_argument("--refs", default=None, help="optional 'utt_id ref_text' file")
    p_ingest.add_argument("--out", required=True, help="canonical utterance JSONL output path")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_score = sub.add_parser("score", help="compute s1-s4 (+ CER if refs present)")
    p_score.add_argument("--instances", required=True, help="canonical utterance JSONL")
    p_score.add_argument("--scores", default=None, help="comma-separated score names (informational)")
    p_score.add_argument("--numeral-policy", default="keep",
                          choices=["keep", "digits-to-cjk", "cjk-to-digits"])
    p_score.add_argument("--out", required=True)
    p_score.set_defaults(func=_cmd_score)

    p_cal = sub.add_parser("calibrate", help="G1 (LTT) + G2 (Mondrian conformal) calibration")
    p_cal.add_argument("--instances", required=True, help="cal-split canonical utterance JSONL")
    p_cal.add_argument(
        "--tune", default=None,
        help="optional speaker-disjoint tune-split JSONL for fitting s5/G2 "
             "(default: auto-split --instances by speaker; see module docstring)",
    )
    p_cal.add_argument("--alpha", type=float, default=0.02)
    p_cal.add_argument("--delta", type=float, default=0.1)
    p_cal.add_argument("--strata", default="duration_tercile",
                        help="comma-separated: duration_tercile,gender")
    p_cal.add_argument("--guarantee", default="ltt", choices=["ltt", "mondrian-ub"])
    p_cal.add_argument("--g1-score", default="s1")
    p_cal.add_argument("--g2-score", default=None)
    p_cal.add_argument("--fit-frac", type=float, default=0.5)
    p_cal.add_argument("--min-stratum-n", type=int, default=_gate.DEFAULT_MIN_STRATUM_N)
    p_cal.add_argument("--n-grid", type=int, default=200,
                        help="LTT lambda-grid size (see asr_gate.ltt.build_lambda_grid)")
    p_cal.add_argument(
        "--min-accept-frac", type=float, default=0.1,
        help="floor on the most-conservative LTT candidate's acceptance fraction "
             "(see asr_gate.ltt.build_lambda_grid docstring for why this matters "
             "for the certificate's power)",
    )
    p_cal.add_argument(
        "--ltt-procedure", default="bonferroni", choices=["bonferroni", "fixed-sequence"],
        help="LTT lambda-selection procedure (see asr_gate.ltt.ltt_certify docstring)",
    )
    p_cal.add_argument(
        "--ltt-pvalue", default="eb", choices=["eb", "hb"],
        help="LTT per-lambda p-value construction (see asr_gate.ltt.ltt_certify docstring)",
    )
    p_cal.add_argument("--numeral-policy", default="keep",
                        choices=["keep", "digits-to-cjk", "cjk-to-digits"])
    p_cal.add_argument("--seed", type=int, default=0)
    p_cal.add_argument("--out", default=None)
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(func=_cmd_calibrate)

    p_apply = sub.add_parser("apply", help="apply a calibrated gate to new (ref-free) utterances")
    p_apply.add_argument("--gate", required=True, help="gate.json from `asr-gate calibrate`")
    p_apply.add_argument("--instances", required=True, help="canonical utterance JSONL")
    p_apply.add_argument("--out", default=None)
    p_apply.add_argument("--json", action="store_true")
    p_apply.set_defaults(func=_cmd_apply)

    p_audit = sub.add_parser("audit", help="excess-AURC + permutation p + Holm across the score family")
    p_audit.add_argument("--instances", required=True, help="canonical utterance JSONL (with refs)")
    p_audit.add_argument("--scores", default=None, help="comma-separated score names (default: s1-s5)")
    p_audit.add_argument("--backbone-field", default=None, help="utterance field naming the backbone")
    p_audit.add_argument("--n-perm", type=int, default=2000)
    p_audit.add_argument("--alpha", type=float, default=0.05)
    p_audit.add_argument("--numeral-policy", default="keep",
                          choices=["keep", "digits-to-cjk", "cjk-to-digits"])
    p_audit.add_argument("--seed", type=int, default=0)
    p_audit.add_argument("--out", default=None)
    p_audit.add_argument("--json", action="store_true")
    p_audit.set_defaults(func=_cmd_audit)

    p_report = sub.add_parser("report", help="compact JSON+Markdown summary (no HTML in the MVP)")
    p_report.add_argument("--audit", default=None, help="audit JSON (from `asr-gate audit`)")
    p_report.add_argument("--gate", default=None, help="gate JSON (from `asr-gate calibrate`)")
    p_report.add_argument("-o", "--output", required=True, help="output .md or .json path")
    p_report.set_defaults(func=_cmd_report)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (_io.SchemaError, _gate.GateError, _audit.AuditError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
