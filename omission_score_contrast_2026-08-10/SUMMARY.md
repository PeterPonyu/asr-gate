# Coupled vs decoupled deploy-valid — ASR landscape CPU contrast

- Date: 2026-08-10
- Landscape: `/home/zeyufu/Desktop/ml-reliability-research/asr-gate/landscape_pulled_2026-07-15`
- Polarity: PASS — s1↔decoupled; hyp_entity_mass↔coupled/conf
- Alphas: [0.05, 0.1, 0.2]
- Wall: 55.2s CPU
- code_sha256: `57528417f16760c703aefbe17ddff63aab08f8364598ef213adb3450e4c6bd78`
- jieba: 0.42.1 vendored under `vendor/`; cache `.jieba_cache/`

## Vacuity definitions

- **oracle_vacuous_frac**: fraction of seeds where oracle flag rate ∈ {0.0, 1.0}
- **deploy_vacuous_frac**: fraction of seeds where deploy flag rate ∈ {0.0, 1.0}
- **Primary stratum (both-arms)**: s1 AND coupled both deploy-nonvacuous (<0.5)
- **Sensitivity (OR)**: either arm deploy-nonvacuous

## Aggregate

- n_pairs (cell×α): 27
- median Δ all pairs (coupled − s1): **+0.000**
- **PRIMARY** both-arms nonvacuous: n=13; median Δ: **0.6**
- **SENSITIVITY** OR nonvacuous: n=27; median Δ: **0.0**
- oracle both-arms nonvacuous: n=10; median Δ: 0.775
- effect direction matches E4 (PRIMARY both-arms): **True**
- D2 claim_strength: **standard**

## AUROC-inverted warning

See `D7_AUROC_DISCLOSURE.md` for per-cell AUROC / inverted flags (CRC accept≥λ polarity).
Cells with AUROC > 0.55 on true-class high-loss ranking are **inverted** for CRC; do not read raw AUROC as deploy-valid success.

## Per-cell deploy-valid

| corpus | backbone | α | s1 deploy% | coupled deploy% | Δ | dep_vac_s1 | dep_vac_c |
|---|---|---:|---:|---:|---:|---:|---:|
| aishell | paraformer | 0.05 | 75.0 | 100.0 | +25.0 | 80% | 0% |
| aishell | paraformer | 0.10 | 100.0 | 100.0 | +0.0 | 80% | 0% |
| aishell | paraformer | 0.20 | 100.0 | 100.0 | +0.0 | 80% | 0% |
| aishell | belle | 0.05 | 0.0 | 35.0 | +35.0 | 0% | 0% |
| aishell | belle | 0.10 | 100.0 | 100.0 | +0.0 | 65% | 0% |
| aishell | belle | 0.20 | 100.0 | 100.0 | +0.0 | 65% | 0% |
| aishell | zipformer | 0.05 | 0.0 | 85.0 | +85.0 | 0% | 0% |
| aishell | zipformer | 0.10 | 90.0 | 100.0 | +10.0 | 75% | 0% |
| aishell | zipformer | 0.20 | 100.0 | 100.0 | +0.0 | 75% | 0% |
| magicdata | paraformer | 0.05 | 0.0 | 75.0 | +75.0 | 0% | 0% |
| magicdata | paraformer | 0.10 | 100.0 | 100.0 | +0.0 | 60% | 0% |
| magicdata | paraformer | 0.20 | 100.0 | 100.0 | +0.0 | 60% | 0% |
| magicdata | belle | 0.05 | 0.0 | 100.0 | +100.0 | 0% | 0% |
| magicdata | belle | 0.10 | 0.0 | 100.0 | +100.0 | 0% | 0% |
| magicdata | belle | 0.20 | 100.0 | 100.0 | +0.0 | 0% | 0% |
| magicdata | zipformer | 0.05 | 0.0 | 100.0 | +100.0 | 0% | 0% |
| magicdata | zipformer | 0.10 | 100.0 | 100.0 | +0.0 | 0% | 0% |
| magicdata | zipformer | 0.20 | 100.0 | 100.0 | +0.0 | 0% | 0% |
| thchs30 | paraformer | 0.05 | 10.0 | 70.0 | +60.0 | 0% | 0% |
| thchs30 | paraformer | 0.10 | 100.0 | 100.0 | +0.0 | 65% | 0% |
| thchs30 | paraformer | 0.20 | 100.0 | 100.0 | +0.0 | 65% | 0% |
| thchs30 | belle | 0.05 | 20.0 | 75.0 | +55.0 | 0% | 0% |
| thchs30 | belle | 0.10 | 0.0 | 80.0 | +80.0 | 20% | 0% |
| thchs30 | belle | 0.20 | 100.0 | 100.0 | +0.0 | 50% | 0% |
| thchs30 | zipformer | 0.05 | 15.0 | 75.0 | +60.0 | 0% | 0% |
| thchs30 | zipformer | 0.10 | 100.0 | 100.0 | +0.0 | 55% | 0% |
| thchs30 | zipformer | 0.20 | 100.0 | 100.0 | +0.0 | 55% | 0% |

Artifacts: `results.json`, `D1_STRATIFIED.md`, `D2_ABLATIONS.md`, `D3_PARETO.md`, `MAPPING.md`, `SELF_OVERLAP.md`.
