# HOI Multi-Model Comparison Visualization — Design Document

Date: 2026-03-06

## Goal

Generate side-by-side visualizations comparing three models on four HOI benchmark tasks to assess training effectiveness. Also surface cases where the reasoning models (SFT, SFT-GRPO) diverge from or reject the input proposals.

## Models

| Label | Folder | Has Thinking |
|-------|--------|--------------|
| Baseline | `results-baseline-qwen3vl-4b-instruct` | No |
| SFT | `results-sft-qwen3vl-4b` | Yes |
| SFT-GRPO | `results-sft-grpo-step140` | Yes |

## Tasks

| Task | Subfolder | Annotation |
|------|-----------|------------|
| HICO Grounding | `hico_ground_sft/` | `hico_ground_test_simplified.json` |
| HICO Action Referring | `hico_action_sft/` | `hico_action_referring_test_simplified.json` |
| SWIG Grounding | `swig_ground_sft/` | `swig_ground_test_simplified.json` |
| SWIG Action Referring | `swig_action_sft/` | `swig_action_referring_test_simplified.json` |

## Specific Result Files

| Task | Baseline | SFT | SFT-GRPO |
|------|----------|-----|----------|
| hico_ground | `hico_ground_qwen3vl_results_20260302_132553.json` | `hico_ground_sft_results_20260228_145909.json` + `_thinking.jsonl` | `hico_ground_sft_results_20260226_144052.json` + `_thinking.jsonl` |
| hico_refer | `hico_action_qwen3vl_results_20260302_132528_per_triplet.json` | `hico_action_sft_results_20260228_145855_per_triplet.json` + `_thinking.jsonl` | `hico_action_sft_results_20260226_144129_per_triplet.json` + `_thinking.jsonl` |
| swig_ground | `swig_ground_qwen3vl_results_20260302_132422.json` | `swig_ground_sft_results_20260228_145937.json` + `_thinking.jsonl` | `swig_ground_sft_results_20260226_143835.json` + `_thinking.jsonl` |
| swig_refer | `swig_action_qwen3vl_results_20260302_132455_per_triplet.json` | `swig_action_sft_results_20260228_145929_per_triplet.json` + `_thinking.jsonl` | `swig_action_sft_results_20260226_143917_per_triplet.json` (thinking inline) |

## Data Sources

| Source | Path | Format |
|--------|------|--------|
| Proposals | `/workspace/hoi-tool-use-checkpoints/test_proposals/{stem}.json` | `{proposals: [{class_name, bbox (pixel), bbox_1000 ([0,1000]), confidence}]}` |
| HICO images | `/workspace/data/hico_20160224_det/images/test2015/` | `.jpg` |
| SWIG images | `/workspace/data/swig_hoi/images_512/` | `.jpg` |
| Annotation (simplified) | `/workspace/Groma/groma_data/benchmarks_simplified/` | `.json` |

## Coordinate Systems

- **Annotation boxes**: pixel coordinates (from simplified JSON)
- **Proposals**: `bbox` = pixel, `bbox_1000` = [0,1000] normalized
- **Model output bboxes**: always [0,1000] → convert to pixel via `x * W/1000, y * H/1000`
- **Baseline grounding**: `[{"pair_id", "person_bbox", "object_bbox"}]` in [0,1000]
- **SFT/GRPO grounding**: `answer` field = `\n`-separated lines, each line = JSON array of `{bbox_2d, label}` per pair (person row + object row alternating)

## Visualization Types

### Type A — 4-Panel Comparison (GT | Baseline | SFT | SFT-GRPO)
- One figure per selected image-action pair
- Each panel: image with colored bboxes overlaid
- Below each panel: predicted action (referring) or pair count + match score (grounding)
- SFT/GRPO panels include: tool_call count, first 3 lines of thinking as caption
- Baseline panel: "No Thinking" label

### Type B — 3-Panel Detail per Model (SFT and GRPO separately)
- Panel 1 — Proposals: all proposals from `test_proposals/` with class label + confidence score
- Panel 2 — Prediction: model output bboxes with labels
- Panel 3 — GT: ground truth boxes
- Saved per model (sft_detail/ and grpo_detail/)

### Type C — Reasoning Trajectory Text Files
- Full `thinking_content` per model (SFT + GRPO)
- Tool_calls sequence shown (zoom_in bbox coords, turn number)
- One `.txt` file per selected sample per task

## Image Selection (10 per task)

### Mandatory Images (MUST include)
- HICO tasks: `HICO_test2015_00009124`, `HICO_test2015_00003584`, `HICO_test2015_00006612`
- SWIG tasks: `checking_24`

### Wrong-Proposal Scoring (≥5 total across all 4 tasks)
See detection section below. Top-scoring samples fill remaining slots.

### Fill
Remaining slots filled with diverse action types, mixing correct and incorrect predictions.

## Wrong-Proposal Detection — Tiered Scoring

Each (file_name, action_object_id) sample is scored 0–10. Samples scoring ≥ 3 are flagged as wrong-proposal cases.

### Signal 1: Empty / None Response (+5 per hit)
- `answer is None`
- `num_pred_pairs == 0`
- Answer string is `"[]"` or empty

### Signal 2: High-Precision Keywords (+4 per match)
Phrases almost exclusively used when model rejects or overrides proposals:
```
"mislabeled as", "mislabelled as", "incorrectly labeled", "incorrectly labelled",
"not in the proposal", "not from the proposal", "no proposal",
"my own bounding box", "estimate my own", "propose my own",
"not detected in", "not present in the proposal",
"I'll use a different", "use different coordinates"
```

### Signal 3: Bbox Divergence (+3 if triggered)
**Grounding tasks only.** For each predicted bbox in [0,1000], compute IoU against every proposal's `bbox_1000`. If max IoU < 0.15 with ALL proposals → model invented its own bbox.
- Guard: skip if `num_pred_pairs == 0` (caught by Signal 1)
- Guard: skip if proposals list is empty

### Signal 4: Zoom Divergence (+3 if triggered)
**Action referring tasks only.** Person+object bboxes are given as input from annotation (in pixel). Scale to [0,1000]. For each `zoom_in` tool call, compute IoU of zoom bbox with both person_box_1000 and object_box_1000. If all zoom_in calls score max(person_iou, object_iou) < 0.05 → model is examining a completely different region.
- Guard: ignore `zoom_out` calls; ignore samples with no tool_calls

### Signal 5: Medium-Confidence Keywords (+2 per match)
Higher-noise phrases; count only if total score < 3 without them:
```
"wrong label", "wrong class", "wrong bounding",
"different region", "different area", "different from the proposal",
"cannot find", "no valid", "not aligned with the proposal",
"incorrectly detected"
```

### Edge Cases

| Edge Case | Handling |
|-----------|----------|
| `answer is None` | Score +5; show "No Response" in visualization panel |
| Empty `thinking_content` | Skip Signals 2 and 5; structural signals still apply |
| Proposals file missing for image | Skip Signal 3; log a warning |
| Empty proposals list | Skip Signal 3; flag as "no proposals available" |
| Model zooms then accepts proposal | Signal 5 only; needs bbox divergence to cross threshold |
| "wrong" in unrelated context | Without bbox divergence, score stays < 3 |
| Very small bboxes | IoU threshold 0.15 (not 0.25) accounts for scale sensitivity |
| Multiple action-object pairs per image | Score per (file_name, action_object_id) pair |
| SWIG file naming | Use `Path(file_name).stem` to build proposal lookup key |
| Baseline (no thinking) | Skip Signals 2, 4, 5; show "No Thinking" label |

## Output Structure

```
/workspace/hoi-benchmarks/visualizations/
├── hico_ground/
│   ├── comparison/              # Type A: 4-panel GT|Baseline|SFT|GRPO
│   ├── sft_detail/              # Type B: 3-panel Proposals|Pred|GT for SFT
│   ├── grpo_detail/             # Type B: 3-panel Proposals|Pred|GT for GRPO
│   ├── reasoning/               # Type C: .txt reasoning trajectories
│   └── selection_manifest.json  # Selected samples + scores + reasons
├── hico_refer/   (same structure)
├── swig_ground/  (same structure)
└── swig_refer/   (same structure)
```

## Color Scheme

| Element | Color |
|---------|-------|
| Person bbox | Red |
| Object bbox | Blue |
| GT pairs | Green |
| Proposals | Orange (with alpha) |
| Predicted pairs | Purple |
| Connecting line (person↔object) | Dashed, same color as pair |

## Implementation

Single script: `/workspace/hoi-benchmarks/generate_visualizations.py`

Run with:
```bash
cd /workspace/hoi-benchmarks
source .venv/bin/activate
uv run python generate_visualizations.py
```

Phases:
1. Load all data into indexed dicts
2. Run wrong-proposal detection → score all samples
3. Select 10 images per task (mandatory + wrong-proposal + fill)
4. Generate all visualizations (Type A, B, C)
5. Save selection manifests
