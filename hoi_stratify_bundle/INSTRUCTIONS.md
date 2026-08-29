# HOI hard/easy stratification — portable bundle

Stratify **existing** HOI eval results into HARD / EASY by proposal difficulty
(`s_ref`), for any number of models, for grounding (Average Recall) and referring
(METEOR / BS-F1) — **without re-running evaluation**. Self-contained (no `verltool`).

`s_ref` = IoU of the best detector (YOLOE) proposal vs the target GT, scored as
`(recall@0.5 + recall@0.75)/2`. It's a property of the sample (proposals + GT), so the
same HARD/EASY label applies to every model → fair comparison.
**EASY = `s_ref ≥ 0.5`** (proposal already good) · **HARD = `s_ref < 0.5`** (proposal misses).

---

## Files
| file | what |
|---|---|
| `paths.json` | **EDIT FIRST** — dataset annotation paths + proposals dir. |
| `models.example.json` | Template registry of models→result-file paths. Copy to `models.json` and edit. |
| `saha_sref.py` | Vendored s_ref math (stdlib only). |
| `compute_sref_cache.py` | **Phase 1** — builds `sref_cache.json` from proposals+GT (run once per test set). |
| `stratify_all_tasks.py` | **Phase 2** — reads the cache + result files, prints/writes stratified tables. |
| `convert_custom_grounding.py` | Converts Claude/GPT-style `predicted_pairs`/`gt_pairs` grounding files into the standard format. |
| `calculate_bertscore.py` | Computes per-sample BERTScore (deberta-v2-xxlarge) for referring BS-F1. |

## Environment
One Python env with: `pycocoevalcap` (METEOR — needs **Java** on PATH, e.g. `openjdk-17-jre-headless`),
`bert_score` + `torch` (only for `calculate_bertscore.py`), and a `numpy`-free stdlib for the rest.
```bash
pip install pycocoevalcap bert_score torch
# METEOR needs Java: apt-get install -y openjdk-17-jre-headless
```

## Run (3 steps)
```bash
# 0. edit paths.json (annotations + proposals dir), then:
python compute_sref_cache.py                          # Phase 1 -> sref_cache.json (once per test set)
cp models.example.json models.json                    # then edit paths to YOUR result files
python stratify_all_tasks.py --models-config models.json --out stratified.json
```
Output: console tables (ALL / EASY / HARD) + `stratified.json` (all numbers). Re-run Phase 2
after editing `models.json`; only re-run Phase 1 if the **test set** changes.

---

## The model registry (`models.json`)
```jsonc
{
  "row_order":  ["qwen8b", "grpo", "claude", "gpt"],         // table row order
  "row_labels": { "qwen8b": "Qwen3-VL-8B", "grpo": "Ours", "claude": "Claude", "gpt": "GPT" },

  "grounding": {                                             // null any (model,dataset) you don't have
    "qwen8b": { "hico": "/path/hico_ground_results.json", "swig": null },
    "claude": { "hico": "/path/hico_ground_normalized.json", "swig": null }
  },
  "referring": {
    "qwen8b": { "hico": null, "swig": { "pred": "/path/..._per_triplet.json",
                                        "bert": "/path/..._per_triplet_bertscore.json" } },
    "claude": { "hico": null, "swig": { "pred": "/path/swig_action_result.json", "bert": null } }
  }
}
```
Keys starting with `_` are ignored (use for comments). A model may appear in only one task.

### What each result file must contain (field names auto-detected; aliases in parentheses)
- **Grounding** — a JSON list (or `{per_sample_results:[...]}`) where each item has:
  `file_name`(`image_file`/`image_path`), `action`, `object`(`object_category`),
  `matches_per_threshold` (`{"0.5":{matched,unmatched_gts},...}` or `"iou_0.50"` keys).
  AR/@.5/@.75 are taken exactly from `matches_per_threshold`. `@s/@m/@l` are reconstructed
  from matched counts + GT object sizes unless the file's predicted boxes (`answer` JSON-lines
  or `generated_text`) re-match the stored AR. A sibling `_metrics.json` (with `ARs/ARm/ARl`)
  gives the exact ALL-bucket size.
- **Referring** — a `_per_triplet.json` list where each item has the GT pair boxes +
  prediction text: `file_name`(`image_file`/`image_path`), `person_bbox`(`person_box`),
  `object_bbox`(`object_box`) (list or `"[..]"` string), `prediction`(`predicted_action`),
  `ground_truth`(`gt_action`). Joined to the cache by the GT pair (robust to row order).
  `bert` = optional per-sample BERTScore JSON (`bertscore_f1` per row); `null` → BS-F1 blank.

### Adding a new model — checklist
1. Add its key to `row_order` + `row_labels`.
2. Put its result paths under `grounding` and/or `referring` (null the cells you lack).
3. **Grounding from raw boxes (e.g. Claude/GPT, API models):** if the file has
   `predicted_pairs`/`gt_pairs` (pixel boxes) instead of `matches_per_threshold`, convert it:
   `python convert_custom_grounding.py IN.json OUT_normalized.json` (prints its AR), then
   point `grounding` at the normalized file. If you only have raw boxes in another schema,
   run the model through your grounding eval harness to produce `matches_per_threshold`.
4. **Referring BS-F1:** `python calculate_bertscore.py --model microsoft/deberta-v2-xxlarge-mnli
   --input <per_triplet.json> --gpu 0` → writes `<...>_bertscore.json`; point `bert` at it.
5. Re-run Phase 2. The console prints `matched=N/total` (referring) and `unmatched=`
   (grounding) per model — if not full, a field-name/box-format mismatch needs a new alias.

---

## Notes / gotchas
- **AR convention:** this tool reports AR = mean recall over IoU 0.5–0.95. Some pipelines call
  AR@0.5 just "AR" — numbers can differ by that definition; @.5 is in the table to cross-check.
- Models scored by different eval scripts can't be perfectly re-matched here, so their
  `@s/@m/@l` are reconstructed (validated to ~0.4 of published ALL); AR/@.5/@.75 stay exact.
- Referring HARD = all "object missed by detector" pairs (single-pair s_ref ∈ {0,0.5,1.0}).
  Treat referring HARD/EASY as **object-perceivability**, not "referring difficulty".
- Every model's ALL-bucket number should reproduce its published `_metrics.json` — the run
  prints `stored_AR` so you can confirm.

---

## Prompt you can give Claude Code on the server
> I have the HOI stratification bundle in this folder. (1) Edit `paths.json` to point at my
> test-set annotation files and YOLOE `test_proposals` dir. (2) Run `python compute_sref_cache.py`.
> (3) Build `models.json` from `models.example.json` with these result files: <list model name →
> hico_ground file + swig_action per_triplet file for each>. For any Claude/GPT-style grounding
> file with `predicted_pairs`, convert it with `convert_custom_grounding.py` first; for referring
> files without a BERTScore sibling, run `calculate_bertscore.py`. (4) Run
> `python stratify_all_tasks.py --models-config models.json --out stratified.json` and show me the
> ALL/EASY/HARD tables. Verify each model's `stored_AR` matches its published AR and that
> `matched=N/total` is full; if a model doesn't join, inspect its field names and add an alias.
