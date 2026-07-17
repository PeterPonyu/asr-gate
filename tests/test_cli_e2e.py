"""End-to-end CLI test: ingest -> score -> calibrate -> apply -> audit ->
report, driven entirely on synthetic data (no network, no GPU).

Asserts (per the verification bar):
  (a) certificate coverage on held-out synthetic data,
  (b) audit correctly flags the informative score and not the random score,
  (c) refusal rules fire: a deliberately undersized Mondrian stratum
      DEFER-ALWAYS, and an injected non-Mandarin hypothesis OOD-REFUSE.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from tests.conftest import make_synthetic_utterances


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "asr_gate.cli", *args],
        capture_output=True, text=True,
    )


def test_full_pipeline_e2e(tmp_path):
    # --- Build a cal pool with a DELIBERATELY undersized gender stratum:
    # keep only 3 female speakers' utterances (a small fraction) so the
    # duration x gender combo lands well under the default min_stratum_n
    # floor (200) while male strata stay large -- a deterministic mix of
    # defer-always and normal strata for the refusal-rule assertion.
    cal_pool = make_synthetic_utterances(
        n=20000, n_speakers=40, seed=100, correlation_strength=1.0, degraded_frac=0.0
    )
    kept_female_speakers = {f"SPK{i:03d}" for i in range(1, 6, 2)}  # 3 of the 20 F speakers
    cal_pool = [
        u for u in cal_pool
        if u["gender"] == "M" or u["speaker_id"] in kept_female_speakers
    ]

    cal_raw_path = tmp_path / "cal_raw.jsonl"
    _write_jsonl(cal_raw_path, cal_pool)

    cal_canonical_path = tmp_path / "cal_canonical.jsonl"
    proc = _run_cli(
        "ingest", "--hyps", str(cal_raw_path), "--format", "custom-schema",
        "--out", str(cal_canonical_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert cal_canonical_path.exists()

    cal_scored_path = tmp_path / "cal_scored.jsonl"
    proc = _run_cli(
        "score", "--instances", str(cal_canonical_path), "--out", str(cal_scored_path),
    )
    assert proc.returncode == 0, proc.stderr
    assert cal_scored_path.exists()

    gate_path = tmp_path / "gate.json"
    proc = _run_cli(
        "calibrate",
        "--instances", str(cal_canonical_path),
        "--alpha", "0.08", "--delta", "0.1",
        "--strata", "duration_tercile,gender",
        "--seed", "100",
        "--n-grid", "200", "--min-accept-frac", "0.15",
        "--out", str(gate_path), "--json",
    )
    assert proc.returncode == 0, proc.stderr
    gate = json.loads(proc.stdout)
    assert gate["g1"]["certified"] is True, gate["g1"]
    lambda_star = gate["g1"]["lambda_star"]
    assert lambda_star is not None

    # (c) refusal rule 1: the undersized female duration-stratum combos
    # must be flagged defer-always; at least one male combo must NOT be.
    defer_always = gate["strata"]["defer_always"]
    female_keys = [k for k in defer_always if k.endswith(":F")]
    male_keys = [k for k in defer_always if k.endswith(":M")]
    assert female_keys, "expected female strata to exist"
    assert all(defer_always[k] for k in female_keys), defer_always
    assert any(not defer_always[k] for k in male_keys), defer_always

    # --- held-out synthetic test set (ref-free apply, but we KEEP the refs
    # locally to check certificate coverage after the fact).
    test_pool = make_synthetic_utterances(
        n=1000, n_speakers=40, seed=101, correlation_strength=1.0, degraded_frac=0.0
    )
    true_cer_by_id = {}
    from asr_gate import cer as _cer_mod
    for u in test_pool:
        true_cer_by_id[u["utt_id"]] = _cer_mod.compute_cer(u["hyp_text"], u["ref_text"])["cer"]

    # Inject one OOD (non-Mandarin) hypothesis.
    ood_utt_id = test_pool[0]["utt_id"]
    test_pool[0] = dict(test_pool[0])
    test_pool[0]["hyp_text"] = "this hypothesis is entirely english text not mandarin at all"
    test_pool[0]["nbest"] = [{"text": test_pool[0]["hyp_text"], "logp": -5.0, "token_logps": None}]

    test_ref_free = [
        {k: v for k, v in u.items() if k != "ref_text"} | {"ref_text": None}
        for u in test_pool
    ]
    test_raw_path = tmp_path / "test_raw.jsonl"
    _write_jsonl(test_raw_path, test_ref_free)
    test_canonical_path = tmp_path / "test_canonical.jsonl"
    proc = _run_cli(
        "ingest", "--hyps", str(test_raw_path), "--format", "custom-schema",
        "--out", str(test_canonical_path),
    )
    assert proc.returncode == 0, proc.stderr

    applied_path = tmp_path / "applied.json"
    proc = _run_cli(
        "apply", "--gate", str(gate_path), "--instances", str(test_canonical_path),
        "--out", str(applied_path), "--json",
    )
    assert proc.returncode == 0, proc.stderr
    applied = json.loads(proc.stdout)

    # (c) refusal rule 2: the injected OOD hypothesis is OOD-REFUSE, a
    # state distinct from DEFER/ACCEPT.
    decisions_by_id = {d["utt_id"]: d for d in applied["decisions"]}
    assert decisions_by_id[ood_utt_id]["action"] == "OOD-REFUSE"
    assert applied["n_ood_refuse"] >= 1

    # (a) certificate coverage: accepted set's TRUE macro-CER <= alpha (+
    # sanity slack; the strict (alpha, delta) statistical guarantee is
    # checked over many resamples in test_ltt.py).
    accepted_ids = [d["utt_id"] for d in applied["decisions"] if d["action"] == "ACCEPT"]
    assert len(accepted_ids) > 0
    accepted_true_cer = [true_cer_by_id[i] for i in accepted_ids if i in true_cer_by_id]
    empirical_macro_cer = sum(accepted_true_cer) / len(accepted_true_cer)
    assert empirical_macro_cer <= gate["alpha"] + 0.05

    # --- audit: informative (s1) vs noise (s3) score, on a fresh draw.
    audit_pool = make_synthetic_utterances(
        n=2000, n_speakers=40, seed=102, correlation_strength=1.0, degraded_frac=0.0
    )
    audit_raw_path = tmp_path / "audit_raw.jsonl"
    _write_jsonl(audit_raw_path, audit_pool)
    audit_canonical_path = tmp_path / "audit_canonical.jsonl"
    proc = _run_cli(
        "ingest", "--hyps", str(audit_raw_path), "--format", "custom-schema",
        "--out", str(audit_canonical_path),
    )
    assert proc.returncode == 0, proc.stderr

    audit_path = tmp_path / "audit.json"
    proc = _run_cli(
        "audit", "--instances", str(audit_canonical_path),
        "--scores", "s1,s3", "--n-perm", "500", "--alpha", "0.05", "--seed", "102",
        "--out", str(audit_path), "--json",
    )
    assert proc.returncode == 0, proc.stderr
    audit_result = json.loads(proc.stdout)
    assert audit_result["holm_family_size"] == 2

    by_score = {r["score"]: r for r in audit_result["results"]}
    assert by_score["s1"]["excess_aurc"] > 0
    assert by_score["s1"]["reject_holm"] is True
    assert by_score["s3"]["reject_holm"] is False

    # --- report: compact Markdown summary.
    report_path = tmp_path / "report.md"
    proc = _run_cli(
        "report", "--audit", str(audit_path), "--gate", str(gate_path),
        "-o", str(report_path),
    )
    assert proc.returncode == 0, proc.stderr
    report_text = report_path.read_text(encoding="utf-8")
    assert "asr-gate report" in report_text
    assert "Certificate (G1, LTT)" in report_text
    assert "Audit (excess-AURC" in report_text


def test_cli_schema_error_exit_code(tmp_path):
    bad_path = tmp_path / "bad.jsonl"
    bad_path.write_text(json.dumps({"utt_id": "x"}) + "\n")
    proc = _run_cli("score", "--instances", str(bad_path), "--out", str(tmp_path / "out.jsonl"))
    assert proc.returncode == 1
    assert "error:" in proc.stderr


def test_decode_paraformer_help_runs_without_funasr():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "orchestration" / "decode_paraformer.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage" in proc.stdout.lower()
