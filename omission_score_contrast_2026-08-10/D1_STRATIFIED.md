# D1 — Stratified deploy-valid contrast

Vacuity: **oracle** = oracle flag rate ∈ {0,1}; **deploy** = deploy flag rate ∈ {0,1}.
Primary stratum: both score arms deploy-nonvacuous (`deploy_vacuous_frac < 0.5` on s1 AND coupled).
Sensitivity: OR deploy-nonvacuous (either arm).

## Headline strata (seed medians aggregated over cell×α pairs)

| stratum | n | median Δ (coupled−s1) |
|---|---:|---:|
| all_pairs | 27 | +0.000 |
| both_arms_deploy_nonvacuous | 13 | +0.600 |
| or_deploy_nonvacuous_sensitivity | 27 | +0.000 |
| both_arms_oracle_nonvacuous | 10 | +0.775 |

## By corpus (all pairs)

| corpus | n | median Δ | n both-arms |
|---|---:|---:|---:|
| aishell | 9 | +0.000 | 2 |
| magicdata | 9 | +0.000 | 7 |
| thchs30 | 9 | +0.000 | 4 |

JSON: `D1_STRATIFIED.json`
