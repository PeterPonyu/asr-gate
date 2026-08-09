# asr-gate report

## Certificate (G1, LTT)

- guarantee: `ltt`
- alpha (target CER): 0.02
- delta (failure prob): 0.1
- certified: **False**
- lambda*: None
- accepted fraction: 0.0000
- n_cal: 3567, n_fit(tune): 3574
- normalizer: `asr-gate-cer-normalizer-v1`

### Mondrian strata

| stratum | n_cal | defer_always | G2 threshold |
|---|---|---|---|
| dur0 | 1189 | False | 0.14988384947607872 |
| dur1 | 1189 | False | 0.1368550883791106 |
| dur2 | 1189 | False | 0.11145790599620058 |

## Audit (excess-AURC, Holm m=2)

- macro-CER: 0.0163
- micro-CER: 0.0161 CI=[0.0145, 0.0178] (speaker-blocked, n_blocks=20)
- clip count: 0

| score | backbone | n | excess-AURC | p | p (Holm) | reject (Holm) |
|---|---|---|---|---|---|---|
| s1 | default | 7184 | 0.0093 | 0.0005 | 0.0010 | True |
| s2 | default | 7184 | 0.0108 | 0.0005 | 0.0010 | True |
