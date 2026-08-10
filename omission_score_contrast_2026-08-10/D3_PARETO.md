# D3 — Deploy flag vs deploy-valid Pareto notes

Tradeoff: lower deploy_flag (abstention) vs higher deploy_valid (coverage at α).
Median deploy_flag_median per score×α across cells:

| score | α | median deploy_flag% | median deploy_valid% |
|---|---:|---:|---:|
| s1 | 0.05 | 8.2 | 0.0 |
| s1 | 0.10 | 0.0 | 100.0 |
| s1 | 0.20 | 0.0 | 100.0 |
| hyp_entity_mass | 0.05 | 16.9 | 75.0 |
| hyp_entity_mass | 0.10 | 4.3 | 100.0 |
| hyp_entity_mass | 0.20 | 4.3 | 100.0 |

**Commentary:** coupled (hyp_entity_mass) tends toward higher deploy_flag at low α
(more abstention on pred-entity stratum) while often matching or exceeding deploy_valid;
decoupled s1 shows inverted AUROC on true-class omission ranking — see SUMMARY D7 pointer.
