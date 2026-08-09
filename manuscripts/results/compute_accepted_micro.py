#!/usr/bin/env python3
"""Post-hoc descriptive companion (red-team m4): accepted-set MICRO-CER for the
main clean Aishell-1 certificate (alpha=2%, 20 reseeds).

The frozen registry (numbers.json <- attainment_table.json) carries the certified
accepted-set MACRO-CER range (1.00-1.53%, mean 1.21%); micro-CER is the field
standard and is reported alongside as a descriptive companion (NOT certified --
the certificate is on macro-CER). This joins each frozen reseed's ACCEPT
decisions to per-utterance edits/ref_len from the frozen test_scored.jsonl and
reports the accepted-set micro-CER (total edits / total ref chars over the
accepted set) across the 20 reseeds. Macro is recomputed as a cross-check
against the frozen 1.00-1.53% / mean 1.21%.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "..", "..", "main_results_2026-07-09")

test = [json.loads(l) for l in open(os.path.join(MAIN, "test_scored.jsonl"))]
by = {r["utt_id"]: r for r in test}

macros, micros, accs = [], [], []
for s in range(20):
    ap = json.load(open(os.path.join(MAIN, f"reseed_{s}", "applied_test.json")))
    acc = [d["utt_id"] for d in ap["decisions"] if d["action"] == "ACCEPT"]
    cers = np.array([by[i]["cer"] for i in acc])
    edits = np.array([by[i].get("edits", np.nan) for i in acc], dtype=float)
    refl = np.array([by[i].get("ref_len", np.nan) for i in acc], dtype=float)
    macros.append(float(cers.mean()))
    micros.append(float(np.nansum(edits) / np.nansum(refl)))
    accs.append(ap["n_accept"] / ap["n"])

out = {
    "description": "post-hoc descriptive companion (m4): accepted-set micro-CER, "
                   "main clean Aishell-1 Paraformer certificate, alpha=2%, 20 reseeds; "
                   "micro is reported not certified",
    "alpha": 0.02, "n_reseeds": 20,
    "acceptance_min": min(accs), "acceptance_max": max(accs), "acceptance_mean": float(np.mean(accs)),
    "accepted_macro_cer_min": min(macros), "accepted_macro_cer_max": max(macros),
    "accepted_macro_cer_mean": float(np.mean(macros)),
    "accepted_micro_cer_min": min(micros), "accepted_micro_cer_max": max(micros),
    "accepted_micro_cer_mean": float(np.mean(micros)),
    "source": "frozen main_results_2026-07-09/reseed_*/applied_test.json joined to "
              "test_scored.jsonl (edits, ref_len)",
}
with open(os.path.join(HERE, "accepted_micro_2026-07-13.json"), "w") as f:
    json.dump(out, f, indent=2)
print("accepted-set MACRO-CER: %.2f-%.2f%% (mean %.2f%%)  [cross-check vs frozen 1.00-1.53/1.21]"
      % (min(macros)*100, max(macros)*100, np.mean(macros)*100))
print("accepted-set MICRO-CER: %.2f-%.2f%% (mean %.2f%%)"
      % (min(micros)*100, max(micros)*100, np.mean(micros)*100))
print("wrote", os.path.join(HERE, "accepted_micro_2026-07-13.json"))
