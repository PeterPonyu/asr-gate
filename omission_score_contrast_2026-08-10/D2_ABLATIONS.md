# D2 — Score ablations

Headline both-arms median Δ (coupled−s1): **0.6**
Auto-soft heuristic tripped: **no** (pred_ent ratio=0.00)

## (a) Binary pred_ent degenerate score (0/1)

- both-arms n=5; median Δ(pred_ent−s1): **0.0**
- all-pairs median Δ: 0.0

## (b) Length-norm vs raw entity count

- hyp_entity_mass = sum(entity mask)/|hyp| (coupled primary)
- hyp_entity_count = sum(entity mask) without /len
- both-arms n=13; median Δ(count−s1): **0.8**

## (c) Scrambled-POS / leave-one-out

- Status: skipped — CPU-only session; scrambled-POS / leave-one-out POS ablation deferred
