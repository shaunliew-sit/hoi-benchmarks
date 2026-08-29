# HOI results stratified by proposal difficulty (s_ref) — all 4 tasks

No re-evaluation — parsed from result files we already have. Difficulty `s_ref`
= SAHA-CF grounding reference score (IoU of best detector proposal vs target GT), the
same quantity the reward uses, recomputed per sample from `test_proposals` + GT.
Buckets: MISS `s_ref≈0` · PARTIAL `0<s_ref<0.5` · EASY `s_ref≥0.5` · **HARD = s_ref<0.5**.

**Pipeline:**
1. `compute_sref_cache.py` (env `verl-tool-env`) → `sref_cache.json` (model-independent).
2. `stratify_all_tasks.py` (env `.venv`, needs pycocoevalcap+Java) → tables + `stratified_all_tasks.json`.

Models/paths live in **`models.json`** (`--models-config` to override). Add a model by
editing that file only — see "Adding a model" below. The stratifier is **model-neutral**:
it auto-handles field-name aliases (`file_name`/`image_file`, `object`/`object_category`,
`prediction`/`predicted_action`), list-or-string boxes, `matches_per_threshold` key
formats (`"0.5"` / `"iou_0.50"`), grounding output as `answer` or `generated_text`, and
joins referring by the GT pair `(file, person_box, object_box)` so row order doesn't matter.

**Validation:** every model's ALL-bucket number reproduces its published `_metrics.json`
(grounding AR/@.5/@.75/@s/@m/@l; referring METEOR; BS-F1).

### Method / honesty notes
- Grounding **AR/@.5/@.75** per bucket come straight from each model's stored
  `matches_per_threshold` — exact for every model.
- Grounding **@s/@m/@l**: exact via re-matching predictions when the re-match reproduces
  that model's stored AR (SAHA SFT/GRPO). For models scored by a *different* eval
  (Qwen base, InternVL3) the re-match can't reproduce them, so @s/@m/@l is **reconstructed
  from stored matched counts + GT object sizes** (exact for single-pair samples,
  largest-object-first for multi-pair; validated to reproduce published ALL size within
  ~0.4). For those models ALL @s/@m/@l = exact published; EASY/HARD = reconstruction.
- **Referring difficulty is a REUSE** of grounding's proposal s_ref as a per-pair
  "object-perceivability" axis (was the object well-detected) — NOT the referring reward's
  own s_ref. Referring is given the GT region, so read referring HARD/EASY as
  "object hard/easy to detect", not "referring difficulty". Referring HARD = all MISS
  (single-pair s_ref ∈ {0,0.5,1.0}, no PARTIAL band).
- **BS-F1** = deberta-v2-xxlarge-mnli raw F1 ×100 (matches the paper table). Blank if no
  per-sample bertscore file is given. **LLM-Judge** not implemented yet (blank).
- Cross-model caveat: base/InternVL3/Groma were scored by different eval scripts than the
  SAHA rows, so cross-model comparisons carry a small eval-convention confound (already
  present in the paper table). The stratification only buckets each model's own results.

---

## GROUNDING — AR / @.5 / @.75 / @s / @m / @l (%)

### ALL
| model | HICO-Ground | SWIG-Ground |
|---|---|---|
| Qwen3-VL-8B | 30.38 / 50.40 / 30.62 / 3.76 / 19.21 / 37.88 | 38.32 / 51.64 / 39.61 / 7.96 / 25.28 / 41.08 |
| +proposal | — / — / — / — / — / — | — / — / — / — / — / — |
| +proposal+SFT | 30.04 / 52.94 / 29.50 / 3.39 / 18.66 / 37.62 | 28.26 / 42.49 / 28.86 / 8.13 / 22.04 / 29.74 |
| Full (+SFT+GRPO) | 31.03 / 54.94 / 30.36 / 3.58 / 19.31 / 38.83 | 29.05 / 43.39 / 29.81 / 8.09 / 21.25 / 30.78 |
| InternVL3-38B | — / — / — / — / — / — | 26.25 / 37.84 / 27.29 / 8.57 / 10.39 / 26.36 |
| Groma-Qwen | — / — / — / — / — / — | — / — / — / — / — / — |

### EASY
| model | HICO-Ground | SWIG-Ground |
|---|---|---|
| Qwen3-VL-8B | 37.93 / 61.46 / 38.75 / 8.76 / 25.31 / 43.06 | 41.98 / 56.04 / 43.51 / 14.90 / 29.45 / 43.88 |
| +proposal | — / — / — / — / — / — | — / — / — / — / — / — |
| +proposal+SFT | 37.86 / 65.75 / 37.53 / 9.78 / 25.98 / 42.72 | 31.45 / 47.06 / 32.19 / 18.97 / 27.77 / 32.07 |
| Full (+SFT+GRPO) | 38.98 / 67.78 / 38.57 / 10.24 / 26.86 / 43.94 | 32.32 / 48.00 / 33.25 / 19.02 / 26.69 / 33.19 |
| InternVL3-38B | — / — / — / — / — / — | 28.77 / 41.09 / 30.00 / 12.06 / 21.77 / 29.85 |
| Groma-Qwen | — / — / — / — / — / — | — / — / — / — / — / — |

### HARD
| model | HICO-Ground | SWIG-Ground |
|---|---|---|
| Qwen3-VL-8B | 5.16 / 13.42 / 3.45 / 1.23 / 4.50 / 8.01 | 7.91 / 15.08 / 7.23 / 2.67 / 9.46 / 8.40 |
| +proposal | — / — / — / — / — / — | — / — / — / — / — / — |
| +proposal+SFT | 3.90 / 10.13 / 2.65 / 0.67 / 2.58 / 6.92 | 1.78 / 4.54 / 1.13 / 0.14 / 1.17 / 2.37 |
| Full (+SFT+GRPO) | 4.44 / 12.04 / 2.90 / 0.76 / 2.72 / 8.05 | 1.88 / 5.11 / 1.23 / 0.04 / 1.38 / 2.46 |
| InternVL3-38B | — / — / — / — / — / — | 5.33 / 10.87 / 4.73 / 1.55 / 7.63 / 5.22 |
| Groma-Qwen | — / — / — / — / — / — | — / — / — / — / — / — |

(+proposal grounding not evaluated; InternVL3 HICO grounding only a partial dump; Groma has no grounding results → `null` in `models.json`.)

---

## REFERRING — METEOR / LLM-Judge / BS-F1 (%)

### ALL
| model | HICO-Refer | SWIG-Refer |
|---|---|---|
| Qwen3-VL-8B | 25.01 / — / 81.84 | 19.08 / — / 77.31 |
| +proposal | 25.61 / — / 82.63 | 18.12 / — / 75.61 |
| +proposal+SFT | 28.51 / — / 84.09 | 16.35 / — / 75.05 |
| Full (+SFT+GRPO) | 29.66 / — / 84.73 | 19.36 / — / 77.90 |
| InternVL3-38B | 25.01 / — / 82.01 | 20.54 / — / 78.59 |
| Groma-Qwen | 8.85 / — / — | 7.11 / — / — |

### EASY
| model | HICO-Refer | SWIG-Refer |
|---|---|---|
| Qwen3-VL-8B | 25.78 / — / 82.29 | 19.06 / — / 77.26 |
| +proposal | 26.35 / — / 83.14 | 18.16 / — / 75.61 |
| +proposal+SFT | 29.43 / — / 84.72 | 16.65 / — / 75.26 |
| Full (+SFT+GRPO) | 30.54 / — / 85.39 | 19.74 / — / 78.19 |
| InternVL3-38B | 25.58 / — / 82.51 | 20.62 / — / 78.61 |
| Groma-Qwen | 9.60 / — / — | 7.25 / — / — |

### HARD
| model | HICO-Refer | SWIG-Refer |
|---|---|---|
| Qwen3-VL-8B | 21.49 / — / 79.64 | 19.24 / — / 77.79 |
| +proposal | 22.25 / — / 80.12 | 17.83 / — / 75.60 |
| +proposal+SFT | 24.29 / — / 81.02 | 13.85 / — / 73.32 |
| Full (+SFT+GRPO) | 25.63 / — / 81.55 | 16.17 / — / 75.47 |
| InternVL3-38B | 22.36 / — / 79.60 | 19.87 / — / 78.39 |
| Groma-Qwen | 5.37 / — / — | 5.98 / — / — |

---

## Reading (SAHA rows)

**GRPO − SFT is positive in every task, dataset, and bucket** = the clean RL-contribution
claim (HICO-G ALL +0.99/EASY +1.12/HARD +0.53; SWIG-G +0.79/+0.88/+0.10; HICO-R EASY +1.11
MET/+0.67 BS, HARD +1.34/+0.53; SWIG-R EASY +3.09/+2.93, HARD +2.32/+2.15).

**SWIG-Ground below base** is an SFT-stage + proposal-dependence artifact concentrated in
EASY (89% of data: base 41.98 → SFT 31.45). Mechanism = MISS/HARD collapse (base
freehand-grounds AR≈8, proposal-conditioned drops to ≈1.8). Do not claim GRPO beats raw
base on HARD (base wins HARD on SWIG-G and SWIG-R; it has no proposal-copy handicap).

**Baselines for context:** InternVL3-38B referring is competitive (SWIG-R 20.54 MET beats
all SAHA rows; HICO-R 25.01 ≈ Qwen base) but its grounding is weak (SWIG-G 26.25). Groma-Qwen
referring is far behind (METEOR 7–9). These are different eval scripts → treat cross-model
gaps as indicative, not exact.

---

## Adding a model (including Claude / GPT)

Edit `models.json` only: add the key to `row_order` + `row_labels`, then its paths.
A model can appear in only one task; null any cell you don't have.

```jsonc
"row_order":  ["base","sft","grpo","internvl3","groma","claude","gpt"],
"row_labels": { ..., "claude": "Claude", "gpt": "GPT-4o" },
"referring": { ...,
  "claude": {"hico": {"pred": "/path/hico_..._per_triplet.json",
                      "bert": "/path/hico_..._per_triplet_bertscore.json"},
             "swig": {"pred": "...", "bert": "..."}} },
"grounding": { ..., "claude": {"hico": "/path/...json", "swig": "/path/...json"} }
```

**What each task needs from a new model (incl. API models like Claude/GPT):**

- **Referring** (easy — text task): a `per_triplet.json` list where each item has the GT
  pair boxes + the prediction text. Required fields (aliases auto-detected):
  `file_name`|`image_file`, `person_bbox`|`person_box`, `object_bbox`|`object_box`
  (list or `"[..]"` string), `prediction`|`predicted_action`, `ground_truth`|`gt_action`.
  For **BS-F1**, run `calculate_bertscore.py --model microsoft/deberta-v2-xxlarge-mnli
  --input <per_triplet.json>` and point `bert` at the output; leave `bert: null` to get
  METEOR only (BS-F1 blank). METEOR is computed by the stratifier.

- **Grounding** (needs IoU matching): the stratifier reads `matches_per_threshold` per
  sample (it does NOT match raw boxes itself unless they reproduce a stored AR). So a new
  model must be run through the **grounding eval harness** (`eval_*_ground_*_qwen3vl.py`)
  to produce a result JSON with `file_name`, `action`, `object`(/`object_category`),
  `matches_per_threshold`, and ideally a sibling `_metrics.json` (for exact ALL @s/@m/@l).
  AR/@.5/@.75 are then exact; @s/@m/@l reconstructed unless the model's `answer`/
  `generated_text` re-matches its stored AR.

After editing, just re-run `.venv/bin/python stratify_all_tasks.py`. The s_ref cache is
model-independent — only rebuild it (`compute_sref_cache.py`) if the test set changes.
The run prints `matched=N/total` per referring model and `unmatched=` per grounding model —
check these are full (a low count means a field-name/box-format mismatch to investigate).
