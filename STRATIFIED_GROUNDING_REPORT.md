# Grounding results stratified by proposal difficulty (s_ref) — HARD vs EASY

**No re-evaluation.** Difficulty `s_ref` = the SAHA-CF grounding reference score (IoU of the
best detector proposal vs target GT, the *same* quantity the reward uses), recomputed per
test sample from `test_proposals` + GT. AR then recomputed per bucket from the existing
per-sample `matches_per_threshold` using the eval's exact micro formula.

**Validation:** recomputed ALL-bucket AR reproduces every published number exactly
(HICO 30.38/30.04/31.03, SWIG 38.32/28.26/29.05), 100% sample join, 0 dup keys.

Buckets: `MISS` = s_ref≈0 (detector totally misses GT) · `PARTIAL` = 0<s_ref<0.5 ·
`EASY` = s_ref≥0.5 · **HARD = MISS+PARTIAL (s_ref<0.5)**.

## Difficulty distribution (test set)
| dataset | n | EASY (s_ref≥0.5) | HARD (s_ref<0.5) | of which MISS (s_ref≈0) |
|---|---|---|---|---|
| HICO | 20,028 | 17,779 (88.8%) | 2,249 (11.2%) | 1,470 |
| SWIG | 19,333 | 17,279 (89.4%) | 2,054 (10.6%) | 2,023 (≈all hard is total-miss) |

## HICO grounding — AR (%) by bucket
| bucket | n | base 8B | +prop+SFT | Full (GRPO) | Δ GRPO−base | Δ GRPO−SFT |
|---|---|---|---|---|---|---|
| ALL | 20,028 | 30.38 | 30.04 | **31.03** | +0.64 | +0.99 |
| EASY | 17,779 | 37.93 | 37.86 | **38.98** | +1.05 | +1.12 |
| HARD | 2,249 | 5.16 | 3.90 | 4.44 | −0.72 | +0.53 |
| MISS | 1,470 | 6.12 | 1.82 | 1.96 | −4.16 | +0.14 |

## SWIG grounding — AR (%) by bucket
| bucket | n | base 8B | +prop+SFT | Full (GRPO) | Δ GRPO−base | Δ GRPO−SFT |
|---|---|---|---|---|---|---|
| ALL | 19,333 | 38.32 | 28.26 | 29.05 | **−9.27** | +0.79 |
| EASY | 17,279 | 41.98 | 31.45 | 32.32 | −9.66 | +0.88 |
| HARD | 2,054 | 7.91 | 1.78 | 1.88 | −6.03 | +0.10 |
| MISS | 2,023 | 7.97 | 1.80 | 1.87 | −6.10 | +0.06 |

## Why SWIG-Ground is below the raw base — the valid reason

1. **It is an SFT-stage regression, not a GRPO regression.** GRPO *improves* over its own
   input (SFT) in **every bucket on both datasets** (SWIG ALL +0.79, EASY +0.88, HARD +0.10).
   The −9.27 vs base is inherited from the proposal+SFT stages (base 38.32 → +proposal 35.78
   → +SFT 28.26), then GRPO recovers +0.79. The correct ablation for "effect of GRPO" is
   **GRPO − SFT, which is positive everywhere.**

2. **The loss is proposal-dependence, exposed by the MISS bucket.** When the detector misses
   the GT (s_ref≈0), the raw base model freehand-grounds from the image and scores AR≈8,
   but the proposal-conditioned SFT/GRPO model collapses to AR≈1.8 — it was taught to trust/
   copy proposals, so a missing proposal sinks it. SWIG's hard cases are *almost entirely*
   total-misses (2,023 MISS vs 31 PARTIAL), because SWIG objects are open-vocabulary/rare and
   YOLOE either nails them or misses completely. So SWIG pays the proposal-dependence penalty
   far harder than HICO.

3. **The penalty also reaches EASY on SWIG, which is why the aggregate tanks.** EASY is ~89%
   of the set, so it dominates ALL. On SWIG EASY the proposal+SFT model (31.45) trails base
   (41.98) by −10.5 — SFT on the combined HOI mix degraded SWIG's fine-grained grounding even
   where the proposal is good. On HICO EASY the SFT was flat (37.93→37.86), so GRPO's easy
   gain (+1.05) lifts the Full model *above* base. That single difference — SFT preserves HICO
   EASY but damages SWIG EASY — is the entire reason HICO ends net-positive and SWIG net-negative.

## Honest framing for the paper
- Report **GRPO vs SFT** as the contribution of the RL stage: positive in every bucket,
  both datasets, strongest on the headline AR@0.5.
- Do **not** claim GRPO beats the raw base on hard cases — it does not (base wins HARD on both,
  because base has no proposal-copy handicap). The known SFT copy/override habit caps hard-case
  tool gain (see memory: saha-cf-tool-collapse-root-cause).
- The SWIG base→Full drop is a **proposal+SFT artifact on rare-object grounding**, isolatable
  to (a) total-miss proposals + (b) SFT damage to SWIG EASY — both fixable by the planned SFT
  trace regen (anti-override + derive-from-crop), not by the reward.
