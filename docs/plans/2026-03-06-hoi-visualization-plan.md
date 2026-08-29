# HOI Multi-Model Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate side-by-side visualizations comparing Baseline, SFT, and SFT-GRPO models across four HOI tasks, with reasoning trajectory files and wrong-proposal case detection.

**Architecture:** Single script `generate_visualizations.py` with modular helpers. Data is loaded once into indexed dicts keyed by `(file_name, action_object_id)` for grounding and `triplet_id` for action referring. Wrong-proposal detection scores all samples before image selection. Three visualization types are generated per task: Type A (4-panel comparison), Type B (3-panel proposals|pred|GT per SFT/GRPO), Type C (reasoning text files). No models are loaded — pure data processing and matplotlib/PIL rendering.

**Tech Stack:** Python 3.11, matplotlib 3.10, Pillow 12, numpy 2.4 — already in `.venv`. Run all Python with `uv run python` or `source .venv/bin/activate && python`.

**Design doc:** `/workspace/hoi-benchmarks/docs/plans/2026-03-06-hoi-visualization-design.md`

---

## Constants and File Paths Reference

```
REPO = /workspace/hoi-benchmarks
PROPOSALS_DIR = /workspace/hoi-tool-use-checkpoints/test_proposals/
HICO_IMAGES = /workspace/data/hico_20160224_det/images/test2015/
SWIG_IMAGES = /workspace/data/swig_hoi/images_512/
ANNOT_DIR = /workspace/Groma/groma_data/benchmarks_simplified/
OUTPUT = /workspace/hoi-benchmarks/visualizations/
```

**Result files (latest/canonical per model per task):**

| Task | Model | JSON | Thinking |
|------|-------|------|----------|
| hico_ground | Baseline | `results-baseline-qwen3vl-4b-instruct/hico_ground_qwen3vl_instruct/hico_ground_qwen3vl_results_20260302_132553.json` | none |
| hico_ground | SFT | `results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909.json` | `...145909_thinking.jsonl` |
| hico_ground | GRPO | `results-sft-grpo-step140/hico_ground_sft/hico_ground_sft_results_20260226_144052.json` | `...144052_thinking.jsonl` |
| hico_refer | Baseline | `results-baseline-qwen3vl-4b-instruct/hico_action_qwen3vl_instruct/hico_action_qwen3vl_results_20260302_132528_per_triplet.json` | none |
| hico_refer | SFT | `results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_per_triplet.json` | `...145855_thinking.jsonl` |
| hico_refer | GRPO | `results-sft-grpo-step140/hico_action_sft/hico_action_sft_results_20260226_144129_per_triplet.json` | `...144129_thinking.jsonl` |
| swig_ground | Baseline | `results-baseline-qwen3vl-4b-instruct/swig_ground_qwen3vl_instruct/swig_ground_qwen3vl_results_20260302_132422.json` | none |
| swig_ground | SFT | `results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937.json` | `...145937_thinking.jsonl` |
| swig_ground | GRPO | `results-sft-grpo-step140/swig_ground_sft/swig_ground_sft_results_20260226_143835.json` | `...143835_thinking.jsonl` |
| swig_refer | Baseline | `results-baseline-qwen3vl-4b-instruct/swig_action_qwen3vl_instruct/swig_action_qwen3vl_results_20260302_132455_per_triplet.json` | none |
| swig_refer | SFT | `results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_per_triplet.json` | `...145929_thinking.jsonl` |
| swig_refer | GRPO | `results-sft-grpo-step140/swig_action_sft/swig_action_sft_results_20260226_143917_per_triplet.json` | thinking inline in per_triplet |

---

## Task 1: Output Directory Setup + Config Module

**Files:**
- Create: `generate_visualizations.py` (skeleton with config constants)

**Step 1: Create the script skeleton with all constants**

```python
#!/usr/bin/env python3
"""
HOI Multi-Model Comparison Visualization Generator.
Run with: source .venv/bin/activate && python generate_visualizations.py
"""
import json
import re
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path("/workspace/hoi-benchmarks")
PROPOSALS_DIR = Path("/workspace/hoi-tool-use-checkpoints/test_proposals")
HICO_IMAGES = Path("/workspace/data/hico_20160224_det/images/test2015")
SWIG_IMAGES = Path("/workspace/data/swig_hoi/images_512")
ANNOT_DIR = Path("/workspace/Groma/groma_data/benchmarks_simplified")
OUTPUT_DIR = REPO / "visualizations"

# ── Result file paths ────────────────────────────────────────────────────────
RESULTS = {
    "hico_ground": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/hico_ground_qwen3vl_instruct/hico_ground_qwen3vl_results_20260302_132553.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/hico_ground_sft/hico_ground_sft_results_20260226_144052.json",
        "grpo_think":REPO / "results-sft-grpo-step140/hico_ground_sft/hico_ground_sft_results_20260226_144052_thinking.jsonl",
    },
    "hico_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/hico_action_qwen3vl_instruct/hico_action_qwen3vl_results_20260302_132528_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/hico_action_sft/hico_action_sft_results_20260226_144129_per_triplet.json",
        "grpo_think":REPO / "results-sft-grpo-step140/hico_action_sft/hico_action_sft_results_20260226_144129_thinking.jsonl",
    },
    "swig_ground": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_ground_qwen3vl_instruct/swig_ground_qwen3vl_results_20260302_132422.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/swig_ground_sft/swig_ground_sft_results_20260226_143835.json",
        "grpo_think":REPO / "results-sft-grpo-step140/swig_ground_sft/swig_ground_sft_results_20260226_143835_thinking.jsonl",
    },
    "swig_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_action_qwen3vl_instruct/swig_action_qwen3vl_results_20260302_132455_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/swig_action_sft/swig_action_sft_results_20260226_143917_per_triplet.json",
        "grpo_think": None,  # thinking is inline in per_triplet for GRPO swig_refer
    },
}

ANNOT_FILES = {
    "hico_ground": ANNOT_DIR / "hico_ground_test_simplified.json",
    "hico_refer":  ANNOT_DIR / "hico_action_referring_test_simplified.json",
    "swig_ground": ANNOT_DIR / "swig_ground_test_simplified.json",
    "swig_refer":  ANNOT_DIR / "swig_action_referring_test_simplified.json",
}

# ── Mandatory images (MUST appear in visualizations) ────────────────────────
MANDATORY = {
    "hico_ground": ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "hico_refer":  ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "swig_ground": ["checking_24"],
    "swig_refer":  ["checking_24"],
}

# ── Scoring thresholds ───────────────────────────────────────────────────────
WRONG_PROPOSAL_THRESHOLD = 3   # samples scoring >= this are "wrong proposal" cases
BBOX_DIVERGE_IOU_THRESH   = 0.15  # predicted bbox vs proposals
ZOOM_DIVERGE_IOU_THRESH   = 0.05  # zoom_in vs given person/object box
IMAGES_PER_TASK           = 10

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "person":   "#FF4444",   # red
    "object":   "#4444FF",   # blue
    "gt":       "#22AA22",   # green
    "proposal": "#FF8800",   # orange
    "pred":     "#AA22FF",   # purple
    "baseline": "#00AAAA",   # teal
}
```

**Step 2: Create output subdirectories**

```python
def setup_output_dirs():
    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        for subdir in ["comparison", "sft_detail", "grpo_detail", "reasoning"]:
            (OUTPUT_DIR / task / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Output dirs created under {OUTPUT_DIR}")
```

**Step 3: Verify all source files exist**

```python
def verify_paths():
    missing = []
    for task, paths in RESULTS.items():
        for key, p in paths.items():
            if p is not None and not Path(p).exists():
                missing.append(f"  [{task}][{key}]: {p}")
    for task, p in ANNOT_FILES.items():
        if not Path(p).exists():
            missing.append(f"  [annot][{task}]: {p}")
    if missing:
        print("MISSING FILES:")
        for m in missing:
            print(m)
        raise FileNotFoundError("Fix missing files before continuing")
    print("All source files verified OK")
```

**Step 4: Add `if __name__ == '__main__'` stub and run**

```python
if __name__ == "__main__":
    setup_output_dirs()
    verify_paths()
    print("Setup complete.")
```

**Step 5: Run to verify**

```bash
cd /workspace/hoi-benchmarks
source .venv/bin/activate
python generate_visualizations.py
```

Expected output:
```
Output dirs created under /workspace/hoi-benchmarks/visualizations
All source files verified OK
Setup complete.
```

**Step 6: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add visualization script skeleton with config and path verification"
```

---

## Task 2: Coordinate Utils + Answer Parser

**Files:**
- Modify: `generate_visualizations.py` — add utility functions after config constants

**Step 1: Add coordinate conversion utilities**

```python
# ── Coordinate utilities ─────────────────────────────────────────────────────

def scale_to_pixel(bbox_1000, width, height):
    """Convert [0,1000]-normalized bbox to pixel coords. Returns [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox_1000
    return [
        max(0, int(x1 * width / 1000)),
        max(0, int(y1 * height / 1000)),
        min(width,  int(x2 * width / 1000)),
        min(height, int(y2 * height / 1000)),
    ]

def scale_to_1000(bbox_pixel, width, height):
    """Convert pixel bbox to [0,1000]-normalized coords."""
    x1, y1, x2, y2 = bbox_pixel
    return [
        int(x1 * 1000 / max(width, 1)),
        int(y1 * 1000 / max(height, 1)),
        int(x2 * 1000 / max(width, 1)),
        int(y2 * 1000 / max(height, 1)),
    ]

def bbox_iou(a, b):
    """Compute IoU between two bboxes [x1,y1,x2,y2] (any consistent scale)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))
    area_b = max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
```

**Step 2: Add grounding answer parsers (SFT/GRPO format and Baseline format)**

```python
# ── Answer parsers ────────────────────────────────────────────────────────────

def parse_ground_answer_sft(answer, width, height):
    """
    Parse SFT/GRPO grounding answer into list of (person_bbox_px, object_bbox_px).
    Answer format: newline-separated lines, each line is a JSON array of {bbox_2d, label}
    representing one pair (person entry + object entry).
    Returns [] on None or unparseable input.
    """
    if not answer:
        return []
    pairs = []
    for line in answer.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            objs = json.loads(line)
            if not isinstance(objs, list):
                continue
            # Find person and non-person object in this pair line
            person_entry = next(
                (o for o in objs if isinstance(o, dict) and
                 o.get('label', '').lower() in ('person', 'human', 'man', 'woman', 'player', 'rider', 'athlete')),
                None
            )
            obj_entry = next(
                (o for o in objs if isinstance(o, dict) and o is not person_entry
                 and 'bbox_2d' in o),
                None
            )
            # Fallback: if no person found, take first and second
            if person_entry is None and len(objs) >= 2:
                person_entry, obj_entry = objs[0], objs[1]
            if person_entry and obj_entry:
                p_bb = person_entry.get('bbox_2d')
                o_bb = obj_entry.get('bbox_2d')
                if p_bb and o_bb and len(p_bb) == 4 and len(o_bb) == 4:
                    pairs.append((
                        scale_to_pixel(p_bb, width, height),
                        scale_to_pixel(o_bb, width, height),
                    ))
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
    return pairs


def parse_ground_answer_baseline(generated_text, width, height):
    """
    Parse baseline grounding answer: [{"pair_id":1,"person_bbox":[...],"object_bbox":[...]}]
    Bboxes are in [0,1000] normalized coords.
    Returns [] on None or unparseable input.
    """
    if not generated_text:
        return []
    # Strip markdown code fences if present
    text = re.sub(r'```[a-z]*', '', generated_text).strip()
    # Find outermost JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        pairs = []
        for item in data:
            if not isinstance(item, dict):
                continue
            p_bb = item.get('person_bbox')
            o_bb = item.get('object_bbox')
            if p_bb and o_bb and len(p_bb) == 4 and len(o_bb) == 4:
                pairs.append((
                    scale_to_pixel(p_bb, width, height),
                    scale_to_pixel(o_bb, width, height),
                ))
        return pairs
    except (json.JSONDecodeError, TypeError):
        return []


def extract_all_pred_bboxes_1000(answer):
    """
    Extract all predicted bboxes (in [0,1000]) from an SFT/GRPO grounding answer.
    Used for bbox divergence check (no coordinate conversion needed).
    Returns flat list of [x1,y1,x2,y2] lists.
    """
    if not answer:
        return []
    bboxes = []
    for line in answer.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            objs = json.loads(line)
            if not isinstance(objs, list):
                continue
            for o in objs:
                bb = o.get('bbox_2d') if isinstance(o, dict) else None
                if bb and len(bb) == 4:
                    bboxes.append(bb)
        except (json.JSONDecodeError, TypeError):
            continue
    return bboxes
```

**Step 3: Add GT pair extractor from annotation**

```python
def extract_gt_pairs_pixel(annot_entry):
    """
    Extract GT (person_bbox, object_bbox) pairs in pixel coords from annotation entry.
    Format: boxes[gt_box_inds[0]], boxes[gt_box_inds[1]] = pair 1 person, pair 1 object, etc.
    """
    boxes = annot_entry['boxes']
    inds = annot_entry['gt_box_inds']
    pairs = []
    for i in range(0, len(inds) - 1, 2):
        p_bb = boxes[inds[i]]
        o_bb = boxes[inds[i + 1]]
        pairs.append((p_bb, o_bb))
    return pairs


def extract_refer_input_bboxes_pixel(annot_entry):
    """
    For action referring: return (person_bbox_px, object_bbox_px) from annotation.
    These are the INPUT bboxes given to the model.
    """
    boxes = annot_entry['boxes']
    p_bb = boxes[annot_entry['person_box_idx']]
    o_bb = boxes[annot_entry['object_box_idx']]
    return p_bb, o_bb
```

**Step 4: Add a quick smoke test inline**

```python
def _test_utils():
    assert scale_to_pixel([500, 500, 750, 750], 640, 480) == [320, 240, 480, 360]
    assert scale_to_1000([320, 240, 480, 360], 640, 480) == [500, 500, 750, 750]
    a = [0, 0, 100, 100]
    assert bbox_iou(a, a) == 1.0
    assert bbox_iou([0,0,100,100], [200,200,300,300]) == 0.0
    # Partial overlap: two 100x100 boxes sharing 50x50
    assert round(bbox_iou([0,0,100,100], [50,50,150,150]), 4) == round(2500/(20000-2500), 4)
    print("Utils tests passed")
```

Add `_test_utils()` call inside `if __name__ == '__main__':` and run.

**Step 5: Run**

```bash
python generate_visualizations.py
```

Expected: `Utils tests passed` printed.

**Step 6: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add coordinate utils and answer parsers with smoke tests"
```

---

## Task 3: Data Loading Layer

**Files:**
- Modify: `generate_visualizations.py` — add data loading functions

**Step 1: Proposal loader**

```python
# ── Data loaders ──────────────────────────────────────────────────────────────

def load_proposals(image_stem):
    """
    Load proposals for an image by its stem (filename without extension).
    Returns list of {class_name, bbox (pixel), bbox_1000, confidence} or [] if missing.
    """
    p = PROPOSALS_DIR / f"{image_stem}.json"
    if not p.exists():
        return []
    with open(p) as f:
        data = json.load(f)
    return data.get('proposals', [])
```

**Step 2: Thinking JSONL loader**

```python
def load_thinking_jsonl(path):
    """
    Load a thinking JSONL file into a dict keyed by (file_name, action).
    Each line: {"file_name":..., "action":..., "thinking":...}
    Returns {} if path is None or file doesn't exist.
    """
    if path is None or not Path(path).exists():
        return {}
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (obj.get('file_name', ''), obj.get('action', ''))
                result[key] = obj.get('thinking', '')
            except json.JSONDecodeError:
                continue
    return result
```

**Step 3: Grounding result loader**

```python
def load_ground_results(task):
    """
    Load grounding results for all 3 models for a given task (hico_ground or swig_ground).
    Returns dict:
      {
        "baseline": {(file_name, action_object_id): entry, ...},
        "sft":      {(file_name, action_object_id): entry, ...},
        "grpo":     {(file_name, action_object_id): entry, ...},
        "sft_thinking":  {(file_name, action): thinking_str},
        "grpo_thinking": {(file_name, action): thinking_str},
      }
    Baseline entry has keys: file_name, action, object, prompt, thinking_content, generated_text, matches_per_threshold
    SFT/GRPO entry has keys: file_name, action, object, num_gt_pairs, num_pred_pairs, tool_calls, answer, matches_per_threshold
    """
    paths = RESULTS[task]

    def _index(json_path, model):
        idx = {}
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            fn = entry.get('file_name', '')
            # Build action_object_id key; baseline uses action_object_id field if present
            aoid = entry.get('action_object_id', f"{entry.get('action','')}_{entry.get('object','')}")
            idx[(fn, aoid)] = entry
        return idx

    baseline_idx = _index(paths['baseline'], 'baseline')
    sft_idx      = _index(paths['sft'], 'sft')
    grpo_idx     = _index(paths['grpo'], 'grpo')

    return {
        "baseline": baseline_idx,
        "sft":      sft_idx,
        "grpo":     grpo_idx,
        "sft_thinking":  load_thinking_jsonl(paths.get('sft_think')),
        "grpo_thinking": load_thinking_jsonl(paths.get('grpo_think')),
    }
```

**Step 4: Action referring result loader**

```python
def load_refer_results(task):
    """
    Load action referring results for all 3 models for a given task (hico_refer or swig_refer).
    Returns dict:
      {
        "baseline": {triplet_id: entry, ...},
        "sft":      {triplet_id: entry, ...},
        "grpo":     {triplet_id: entry, ...},
        "sft_thinking":  {(file_name, ''): thinking_str},  # JSONL keyed by file_name only
        "grpo_thinking": {(file_name, ''): thinking_str},
      }
    Entry keys: triplet_id, file_name, ground_truth, prediction, tool_calls, exact_match, thinking_content
    """
    paths = RESULTS[task]

    def _index(json_path):
        idx = {}
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            tid = entry.get('triplet_id', len(idx))
            idx[tid] = entry
        return idx

    baseline_idx = _index(paths['baseline'])
    sft_idx      = _index(paths['sft'])
    grpo_idx     = _index(paths['grpo'])

    # For action referring, thinking may be in separate JSONL or inline in per_triplet
    # The per_triplet already has thinking_content field inline; JSONL is supplementary
    sft_thinking  = load_thinking_jsonl(paths.get('sft_think'))
    grpo_thinking = load_thinking_jsonl(paths.get('grpo_think'))

    return {
        "baseline": baseline_idx,
        "sft":      sft_idx,
        "grpo":     grpo_idx,
        "sft_thinking":  sft_thinking,
        "grpo_thinking": grpo_thinking,
    }
```

**Step 5: Annotation loader**

```python
def load_annotation(task):
    """
    Load annotation for a task.
    Grounding: returns list indexed positionally, keyed by (file_name, action_object_id).
    Referring: returns list indexed by triplet_id (positional index = triplet_id).
    Returns (list, dict_index).
    """
    with open(ANNOT_FILES[task]) as f:
        annots = json.load(f)

    if 'ground' in task:
        idx = {}
        for entry in annots:
            fn = entry['file_name']
            aoid = entry.get('action_object_id', f"{entry.get('action','')}_{entry.get('object_category','')}")
            idx[(fn, aoid)] = entry
        return annots, idx
    else:
        # For action referring, triplet_id = positional index
        idx = {i: entry for i, entry in enumerate(annots)}
        return annots, idx
```

**Step 6: Run to ensure loading works without crash**

Add inside `__main__`:
```python
print("Loading hico_ground data...")
hg = load_ground_results("hico_ground")
print(f"  baseline entries: {len(hg['baseline'])}")
print(f"  sft entries:      {len(hg['sft'])}")
print(f"  grpo entries:     {len(hg['grpo'])}")
print("Loading hico_refer data...")
hr = load_refer_results("hico_refer")
print(f"  baseline triplets: {len(hr['baseline'])}")
print("Data loading OK")
```

```bash
python generate_visualizations.py
```

Expected: counts around 20028 for grounding, 33405 for HICO referring.

**Step 7: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add proposal, thinking, and result data loaders"
```

---

## Task 4: Wrong-Proposal Scorer — Signals 1, 2, 5 (Text-Based)

**Files:**
- Modify: `generate_visualizations.py` — add scoring constants and text-signal functions

**Step 1: Add keyword lists**

```python
# ── Wrong-proposal detection keywords ────────────────────────────────────────

HIGH_PRECISION_KEYWORDS = [
    "mislabeled as", "mislabelled as",
    "incorrectly labeled", "incorrectly labelled",
    "not in the proposal", "not from the proposal",
    "no proposal", "no valid proposal",
    "my own bounding box", "estimate my own",
    "propose my own", "my own estimate",
    "not detected in the proposal", "not present in the proposal",
    "i'll use a different", "use different coordinates",
    "not from any proposal",
]

MEDIUM_KEYWORDS = [
    "wrong label", "wrong class", "wrong bounding",
    "different region", "different area",
    "different from the proposal",
    "cannot find", "no valid",
    "not aligned with the proposal",
    "incorrectly detected",
    "wrong detection",
]
```

**Step 2: Add text-signal scorer**

```python
def score_text_signals(thinking_content):
    """
    Score a thinking string for Signals 2 (high-precision) and 5 (medium) keywords.
    Returns (signal2_score, signal5_score, matched_keywords).
    """
    if not thinking_content:
        return 0, 0, []

    text = thinking_content.lower()
    matched = []

    s2 = 0
    for kw in HIGH_PRECISION_KEYWORDS:
        if kw in text:
            s2 += 4
            matched.append(("HIGH", kw))

    s5 = 0
    for kw in MEDIUM_KEYWORDS:
        if kw in text:
            s5 += 2
            matched.append(("MED", kw))

    return s2, s5, matched


def score_empty_signal(entry, is_ground):
    """
    Signal 1: empty/None response.
    For grounding: check answer is None or num_pred_pairs == 0 or answer is '[]'.
    For referring: no equivalent (model always produces a text response).
    Returns score (0 or 5).
    """
    if not is_ground:
        return 0, []
    reasons = []
    if entry.get('answer') is None:
        reasons.append("answer=None")
    if entry.get('num_pred_pairs', -1) == 0:
        reasons.append("num_pred_pairs=0")
    if isinstance(entry.get('answer'), str) and entry['answer'].strip() in ('', '[]'):
        reasons.append("answer='[]'")
    return (5 if reasons else 0), reasons
```

**Step 3: Integrate text signals into a sample scorer stub**

```python
def score_sample(entry, thinking_content, is_ground,
                 annot_entry=None, proposals=None, model=None):
    """
    Compute wrong-proposal score for a single sample.
    Returns (total_score, reasons_dict).
    is_ground: True for grounding tasks, False for action referring.
    annot_entry: annotation dict (needed for Signal 4).
    proposals: list of proposal dicts (needed for Signal 3).
    model: 'baseline', 'sft', or 'grpo'.
    """
    if model == 'baseline':
        return 0, {}   # baseline has no thinking; skip all signals

    score = 0
    reasons = {}

    # Signal 1
    s1, r1 = score_empty_signal(entry, is_ground)
    score += s1
    if r1:
        reasons['signal1_empty'] = r1

    # Signals 2 + 5 (text)
    s2, s5, kw_matches = score_text_signals(thinking_content)
    score += s2
    if s2 > 0:
        reasons['signal2_high_kw'] = [m[1] for m in kw_matches if m[0] == 'HIGH']

    # Signal 3 and 4 added in Task 5
    # Signal 5: apply only if not already high-scoring
    score += s5
    if s5 > 0:
        reasons['signal5_med_kw'] = [m[1] for m in kw_matches if m[0] == 'MED']

    return score, reasons
```

**Step 4: Quick test on a known case**

Add inside `__main__` after data loading:
```python
# Test text scoring on a known mislabel case
test_thinking = "The proposal mislabeled as 'pirate' is clearly a kite. Not in the proposal at all."
s2, s5, kws = score_text_signals(test_thinking)
assert s2 >= 4, f"Expected HIGH keyword hit, got s2={s2}"
print(f"Text signal test passed: s2={s2}, s5={s5}, kws={kws}")
```

```bash
python generate_visualizations.py
```

**Step 5: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add text-signal wrong-proposal scorer (signals 1, 2, 5)"
```

---

## Task 5: Wrong-Proposal Scorer — Signals 3 and 4 (Structural)

**Files:**
- Modify: `generate_visualizations.py` — add bbox divergence and zoom divergence signals to `score_sample`

**Step 1: Add Signal 3 — bbox divergence for grounding**

```python
def signal3_bbox_diverge(entry, proposals):
    """
    Signal 3 (grounding only): predicted bbox has max IoU < BBOX_DIVERGE_IOU_THRESH
    with ALL proposal bboxes_1000.
    Returns (score, reasons).
    """
    if not proposals:
        return 0, []  # no proposals to compare — can't determine divergence

    answer = entry.get('answer')
    if not answer:
        return 0, []  # caught by Signal 1

    pred_bboxes = extract_all_pred_bboxes_1000(answer)
    if not pred_bboxes:
        return 0, []

    prop_bboxes_1000 = [p['bbox_1000'] for p in proposals]
    diverged = []
    for pred_bb in pred_bboxes:
        max_iou = max((bbox_iou(pred_bb, pb) for pb in prop_bboxes_1000), default=0.0)
        if max_iou < BBOX_DIVERGE_IOU_THRESH:
            diverged.append({'pred_bbox': pred_bb, 'max_iou': round(max_iou, 3)})

    if diverged:
        return 3, diverged
    return 0, []
```

**Step 2: Add Signal 4 — zoom divergence for action referring**

```python
def signal4_zoom_diverge(entry, annot_entry):
    """
    Signal 4 (action referring only): ALL zoom_in tool calls target a region
    with IoU < ZOOM_DIVERGE_IOU_THRESH with BOTH the given person AND object boxes.
    Returns (score, reasons).
    """
    if annot_entry is None:
        return 0, []

    tool_calls = entry.get('tool_calls', [])
    zoom_ins = [tc for tc in tool_calls if tc.get('name') == 'zoom_in']
    if not zoom_ins:
        return 0, []

    w = annot_entry.get('width', 1)
    h = annot_entry.get('height', 1)
    boxes = annot_entry.get('boxes', [])

    p_idx = annot_entry.get('person_box_idx')
    o_idx = annot_entry.get('object_box_idx')
    if p_idx is None or o_idx is None or p_idx >= len(boxes) or o_idx >= len(boxes):
        return 0, []

    p_bb_1000 = scale_to_1000(boxes[p_idx], w, h)
    o_bb_1000 = scale_to_1000(boxes[o_idx], w, h)

    all_diverged = True
    diverged_zooms = []
    for tc in zoom_ins:
        zoom_bb = tc.get('bbox', [])
        if len(zoom_bb) != 4:
            continue
        p_iou = bbox_iou(zoom_bb, p_bb_1000)
        o_iou = bbox_iou(zoom_bb, o_bb_1000)
        max_iou = max(p_iou, o_iou)
        if max_iou >= ZOOM_DIVERGE_IOU_THRESH:
            all_diverged = False
            break
        diverged_zooms.append({'zoom_bbox': zoom_bb, 'person_iou': round(p_iou,3), 'object_iou': round(o_iou,3)})

    if all_diverged and diverged_zooms:
        return 3, diverged_zooms
    return 0, []
```

**Step 3: Update `score_sample` to call Signals 3 and 4**

```python
def score_sample(entry, thinking_content, is_ground,
                 annot_entry=None, proposals=None, model=None):
    if model == 'baseline':
        return 0, {}

    score = 0
    reasons = {}

    # Signal 1
    s1, r1 = score_empty_signal(entry, is_ground)
    score += s1
    if r1:
        reasons['signal1_empty'] = r1

    # Signals 2 + 5 (text)
    s2, s5, kw_matches = score_text_signals(thinking_content)
    score += s2
    if s2 > 0:
        reasons['signal2_high_kw'] = [m[1] for m in kw_matches if m[0] == 'HIGH']

    # Signal 3 (grounding only)
    if is_ground:
        s3, r3 = signal3_bbox_diverge(entry, proposals)
        score += s3
        if r3:
            reasons['signal3_bbox_diverge'] = r3

    # Signal 4 (referring only)
    if not is_ground:
        s4, r4 = signal4_zoom_diverge(entry, annot_entry)
        score += s4
        if r4:
            reasons['signal4_zoom_diverge'] = r4

    # Signal 5
    score += s5
    if s5 > 0:
        reasons['signal5_med_kw'] = [m[1] for m in kw_matches if m[0] == 'MED']

    return score, reasons
```

**Step 4: Test Signal 3 on a known diverged case**

```python
# In __main__ test block:
fake_entry = {"answer": '[{"bbox_2d": [800, 800, 900, 900], "label": "person"}]',
              "num_pred_pairs": 1}
fake_proposals = [{"bbox_1000": [0, 0, 100, 100], "class_name": "person", "confidence": 0.9,
                   "bbox": [0,0,64,48]}]
s3, r3 = signal3_bbox_diverge(fake_entry, fake_proposals)
assert s3 == 3, f"Expected s3=3, got {s3}"
print(f"Signal3 test passed: score={s3}, reasons={r3}")

# Test Signal 4 with zoom far from person/object
fake_tc_entry = {
    "tool_calls": [{"name": "zoom_in", "bbox": [800, 800, 1000, 1000]}],
    "thinking_content": ""
}
fake_annot = {"width": 640, "height": 480,
              "boxes": [[10, 10, 50, 50], [60, 60, 100, 100]],
              "person_box_idx": 0, "object_box_idx": 1}
s4, r4 = signal4_zoom_diverge(fake_tc_entry, fake_annot)
assert s4 == 3, f"Expected s4=3, got {s4}"
print(f"Signal4 test passed: score={s4}")
```

```bash
python generate_visualizations.py
```

**Step 5: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add structural wrong-proposal signals 3 (bbox diverge) and 4 (zoom diverge)"
```

---

## Task 6: Score All Samples + Image Selection

**Files:**
- Modify: `generate_visualizations.py` — add `score_all_samples` and `select_images`

**Step 1: Score all samples for a task**

```python
def score_all_samples(task, results, annots_list, annots_idx):
    """
    Score all SFT and GRPO samples for wrong-proposal detection.
    Returns dict: {(file_name, key): {"sft": (score, reasons), "grpo": (score, reasons)}}
    where key is action_object_id for grounding or triplet_id for referring.
    """
    is_ground = 'ground' in task
    scored = {}

    for model in ('sft', 'grpo'):
        result_idx  = results[model]
        think_map   = results[f'{model}_thinking']

        for lookup_key, entry in result_idx.items():
            fn = entry.get('file_name', '')
            image_stem = Path(fn).stem

            # Get annotation entry
            if is_ground:
                annot = annots_idx.get(lookup_key)
            else:
                tid = entry.get('triplet_id', None)
                annot = annots_idx.get(tid) if tid is not None else None

            # Get thinking content
            if is_ground:
                action = entry.get('action', '')
                thinking = think_map.get((fn, action), '')
                if not thinking:
                    thinking = entry.get('thinking_content', '')
            else:
                thinking = entry.get('thinking_content', '')
                if not thinking:
                    action = ''
                    thinking = think_map.get((fn, action), '')

            # Load proposals for structural signals
            proposals = load_proposals(image_stem) if is_ground else None

            score, reasons = score_sample(
                entry=entry,
                thinking_content=thinking,
                is_ground=is_ground,
                annot_entry=annot,
                proposals=proposals,
                model=model,
            )

            composite_key = lookup_key  # (fn, aoid) or triplet_id
            if composite_key not in scored:
                scored[composite_key] = {}
            scored[composite_key][model] = (score, reasons)

    return scored
```

**Step 2: Image selection function**

```python
def select_images_for_task(task, results, annots_list, annots_idx, scored,
                            n=IMAGES_PER_TASK):
    """
    Select n samples for visualization.
    Priority:
      1. Mandatory images (MUST include, up to all their action-object pairs)
      2. Highest-scoring wrong-proposal cases (score >= WRONG_PROPOSAL_THRESHOLD)
      3. Fill remaining with diverse, non-duplicate samples

    Returns list of lookup_keys (grounding: (fn, aoid), referring: triplet_id).
    """
    is_ground = 'ground' in task
    mandatory_stems = MANDATORY.get(task, [])
    selected = []
    selected_set = set()
    selected_stems = set()

    # Helper to get stem from a key
    def get_stem(key):
        if is_ground:
            fn = key[0]
        else:
            entry = results['sft'].get(key) or results['baseline'].get(key) or {}
            fn = entry.get('file_name', '')
        return Path(fn).stem

    # Pass 1: mandatory images — include FIRST action-object pair matching their stem
    for stem in mandatory_stems:
        for key in (results['sft'] if is_ground else results['sft']).keys():
            if get_stem(key) == stem and key not in selected_set:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)
                break  # one entry per mandatory image

    # Pass 2: wrong-proposal cases (score >= threshold, not already selected)
    wrong_cases = []
    for key, model_scores in scored.items():
        max_score = max((v[0] for v in model_scores.values()), default=0)
        if max_score >= WRONG_PROPOSAL_THRESHOLD and key not in selected_set:
            stem = get_stem(key)
            if stem not in selected_stems:
                wrong_cases.append((max_score, key))
    wrong_cases.sort(key=lambda x: -x[0])  # highest score first
    for score, key in wrong_cases:
        if len(selected) >= n:
            break
        selected.append(key)
        selected_set.add(key)
        selected_stems.add(get_stem(key))

    # Pass 3: fill remaining with diverse samples
    if len(selected) < n:
        all_keys = list((results['sft'] if is_ground else results['sft']).keys())
        # Shuffle for diversity (use fixed seed for reproducibility)
        import random
        rng = random.Random(42)
        rng.shuffle(all_keys)
        for key in all_keys:
            if len(selected) >= n:
                break
            stem = get_stem(key)
            if key not in selected_set and stem not in selected_stems:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)

    return selected[:n]
```

**Step 3: Build selection manifest**

```python
def build_selection_manifest(task, selected_keys, scored, results):
    """Build and save selection_manifest.json for a task."""
    is_ground = 'ground' in task
    manifest = []
    for key in selected_keys:
        entry = (results['sft'].get(key) if is_ground
                 else results['sft'].get(key)) or {}
        model_scores = scored.get(key, {})
        manifest.append({
            "key": str(key),
            "file_name": entry.get('file_name', ''),
            "action": entry.get('action', entry.get('ground_truth', '')),
            "sft_score":  model_scores.get('sft', (0, {}))[0],
            "grpo_score": model_scores.get('grpo', (0, {}))[0],
            "sft_reasons":  model_scores.get('sft', (0, {}))[1],
            "grpo_reasons": model_scores.get('grpo', (0, {}))[1],
        })
    out_path = OUTPUT_DIR / task / "selection_manifest.json"
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {out_path}")
    return manifest
```

**Step 4: Test selection on hico_ground**

Add in `__main__`:
```python
print("Testing image selection on hico_ground...")
annots_list, annots_idx = load_annotation("hico_ground")
results = load_ground_results("hico_ground")
scored = score_all_samples("hico_ground", results, annots_list, annots_idx)
selected = select_images_for_task("hico_ground", results, annots_list, annots_idx, scored)
print(f"  Selected {len(selected)} samples")
# Verify mandatory images are present
for stem in MANDATORY["hico_ground"]:
    found = any(stem in str(k) for k in selected)
    assert found, f"Mandatory image {stem} not in selection!"
print(f"  All mandatory images present")
# Count wrong-proposal cases
wp_count = sum(1 for k in selected
               if max((scored.get(k, {}).get(m, (0,{}))[0] for m in ('sft','grpo')), default=0) >= WRONG_PROPOSAL_THRESHOLD)
print(f"  Wrong-proposal cases in selection: {wp_count}")
```

```bash
python generate_visualizations.py
```

**Step 5: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add sample scoring, image selection, and manifest builder"
```

---

## Task 7: Drawing Primitives

**Files:**
- Modify: `generate_visualizations.py` — add drawing utility functions

**Step 1: Image loader**

```python
# ── Drawing utilities ─────────────────────────────────────────────────────────

def load_image(file_name, task):
    """Load PIL image for a given file_name and task (determines image dir)."""
    if 'hico' in task:
        img_path = HICO_IMAGES / file_name
    else:
        img_path = SWIG_IMAGES / file_name
    if not img_path.exists():
        # Try swapping extension
        for ext in ('.jpg', '.jpeg', '.png'):
            alt = img_path.with_suffix(ext)
            if alt.exists():
                return Image.open(alt).convert('RGB')
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert('RGB')
```

**Step 2: BBox drawing function**

```python
def draw_bboxes_on_ax(ax, img_array, bbox_groups, title, title_color='black', fontsize=9):
    """
    Draw an image with colored bounding boxes on a matplotlib Axes.

    bbox_groups: list of dicts, each with:
      {
        'bboxes': [(p_bbox, o_bbox), ...],  # pixel coords [x1,y1,x2,y2]
        'p_color': hex_str,
        'o_color': hex_str,
        'label': str,                        # shown in legend
        'linestyle': '-' or '--',
      }
    Single-box entries use only 'bboxes': [[x1,y1,x2,y2]] and 'color'.
    """
    ax.imshow(img_array)
    ax.set_title(title, color=title_color, fontsize=fontsize, fontweight='bold', pad=4)
    ax.axis('off')

    h, w = img_array.shape[:2]
    legend_patches = []

    for group in bbox_groups:
        bboxes = group.get('bboxes', [])
        p_color = group.get('p_color', COLORS['pred'])
        o_color = group.get('o_color', COLORS['pred'])
        single_color = group.get('color', p_color)
        label = group.get('label', '')
        ls = group.get('linestyle', '-')
        lw = group.get('linewidth', 2.0)

        for item in bboxes:
            if isinstance(item[0], list):
                # (person_bbox, object_bbox) pair
                p_bb, o_bb = item
                for bb, color in [(p_bb, p_color), (o_bb, o_color)]:
                    x1, y1, x2, y2 = bb
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=lw, edgecolor=color, facecolor='none',
                        linestyle=ls
                    )
                    ax.add_patch(rect)
                # Connecting line between centers
                p_cx = (p_bb[0] + p_bb[2]) / 2
                p_cy = (p_bb[1] + p_bb[3]) / 2
                o_cx = (o_bb[0] + o_bb[2]) / 2
                o_cy = (o_bb[1] + o_bb[3]) / 2
                ax.plot([p_cx, o_cx], [p_cy, o_cy], color=p_color, lw=1, ls='--', alpha=0.6)
            else:
                # Single bbox [x1,y1,x2,y2]
                x1, y1, x2, y2 = item
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=lw, edgecolor=single_color, facecolor='none',
                    linestyle=ls
                )
                ax.add_patch(rect)

        if label and bboxes:
            legend_patches.append(patches.Patch(color=single_color or p_color, label=label))

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=6,
                  framealpha=0.7, markerscale=0.8)
```

**Step 3: Text wrap utility for thinking captions**

```python
def wrap_thinking(text, max_lines=4, line_width=60):
    """Truncate and wrap thinking content for display below a visualization panel."""
    if not text:
        return "(no thinking)"
    # Take first paragraph (up to first double newline)
    first_para = text.split('\n\n')[0].replace('\n', ' ').strip()
    wrapped = textwrap.fill(first_para, width=line_width)
    lines = wrapped.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:line_width - 3] + '...'
    return '\n'.join(lines)
```

**Step 4: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add drawing primitives for bbox overlay and text wrapping"
```

---

## Task 8: Type A — 4-Panel Comparison Visualization

**Files:**
- Modify: `generate_visualizations.py` — add `generate_comparison_grounding` and `generate_comparison_referring`

**Step 1: Comparison figure for grounding**

```python
def generate_comparison_grounding(task, key, results, annots_idx, output_dir):
    """
    Generate Type A: 4-panel comparison figure (GT | Baseline | SFT | GRPO) for grounding.
    key: (file_name, action_object_id)
    """
    fn, aoid = key
    annot = annots_idx.get(key)
    if annot is None:
        print(f"  [SKIP] No annotation for {key}")
        return

    img = load_image(fn, task)
    img_arr = np.array(img)
    W, H = img.size

    gt_pairs_px = extract_gt_pairs_pixel(annot)
    action = annot.get('action', aoid)
    obj_cat = annot.get('object_category', '')
    title_info = f"{action} | {obj_cat}"

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{fn}  —  {title_info}", fontsize=10, y=1.01)

    panel_data = [
        ("Ground Truth", "gt",
         [{"bboxes": [[p, o] for p, o in gt_pairs_px],
           "p_color": COLORS["gt"], "o_color": COLORS["gt"], "label": "GT pair", "linestyle": "-"}],
         None),
    ]

    for model, color_key, result_key in [
        ("Baseline", "baseline", "baseline"),
        ("SFT",      "sft",      "sft"),
        ("SFT-GRPO", "grpo",     "grpo"),
    ]:
        entry = results[result_key].get(key, {})
        thinking_map = results.get(f'{result_key}_thinking', {})
        thinking = thinking_map.get((fn, action), '') or entry.get('thinking_content', '')

        if result_key == 'baseline':
            pred_pairs = parse_ground_answer_baseline(
                entry.get('generated_text', ''), W, H)
        else:
            pred_pairs = parse_ground_answer_sft(entry.get('answer'), W, H)

        n_pred = len(pred_pairs)
        n_gt = len(gt_pairs_px)
        matched = entry.get('matches_per_threshold', {}).get('0.5', {}).get('matched', '?')
        subtitle = f"{model}\npred={n_pred}, gt={n_gt}, AR@0.5 matched={matched}"
        if thinking and result_key != 'baseline':
            subtitle += f"\n[{wrap_thinking(thinking, max_lines=2)}]"

        bbox_groups = [{"bboxes": [[p, o] for p, o in pred_pairs],
                        "p_color": COLORS["person"], "o_color": COLORS["object"],
                        "label": "Pred pair", "linestyle": "-"}]

        panel_data.append((subtitle, result_key, bbox_groups, None))

    for ax, (subtitle, _, bbox_groups, _) in zip(axes, panel_data):
        draw_bboxes_on_ax(ax, img_arr, bbox_groups, subtitle, fontsize=7)

    plt.tight_layout()
    safe_name = fn.replace('.jpg', '').replace('.png', '') + f"__{aoid.replace('/', '_')}"
    out_path = output_dir / f"{safe_name}_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")
```

**Step 2: Comparison figure for action referring**

```python
def generate_comparison_referring(task, key, results, annots_idx, output_dir):
    """
    Generate Type A: 4-panel comparison figure (GT | Baseline | SFT | GRPO) for action referring.
    key: triplet_id (int)
    """
    annot = annots_idx.get(key)
    if annot is None:
        print(f"  [SKIP] No annotation for triplet_id={key}")
        return

    fn = annot['file_name']
    img = load_image(fn, task)
    img_arr = np.array(img)
    W, H = img.size
    gt_action = annot.get('gt_action', '')
    p_bb_px, o_bb_px = extract_refer_input_bboxes_pixel(annot)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{fn}  —  GT: {gt_action}", fontsize=10, y=1.01)

    # Person/object input bboxes (same for all panels — always shown)
    input_group = [
        {"bboxes": [p_bb_px], "color": COLORS["person"], "label": "Person (input)", "linewidth": 2.5},
        {"bboxes": [o_bb_px], "color": COLORS["object"],  "label": "Object (input)",  "linewidth": 2.5},
    ]

    # GT panel
    draw_bboxes_on_ax(axes[0], img_arr, input_group,
                      f"Ground Truth\n{gt_action}", fontsize=8)

    for ax, (model_label, result_key) in zip(axes[1:], [
        ("Baseline", "baseline"),
        ("SFT",      "sft"),
        ("SFT-GRPO", "grpo"),
    ]):
        entry = results[result_key].get(key, {})
        thinking = entry.get('thinking_content', '')
        think_map = results.get(f'{result_key}_thinking', {})
        if not thinking:
            thinking = think_map.get((fn, ''), '')

        pred = entry.get('prediction', '—')
        exact = entry.get('exact_match', False)
        n_tools = len(entry.get('tool_calls', []))

        color = '#22AA22' if exact else '#CC2222'
        subtitle = f"{model_label}\nPred: {pred}"
        if model_label != "Baseline":
            subtitle += f"  | tools: {n_tools}"
            subtitle += f"\n[{wrap_thinking(thinking, max_lines=2)}]"

        draw_bboxes_on_ax(ax, img_arr, input_group, subtitle,
                          title_color=color, fontsize=7)

    plt.tight_layout()
    stem = Path(fn).stem
    out_path = output_dir / f"{stem}__triplet{key}_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")
```

**Step 3: Test on one hico_ground sample**

Add in `__main__` (temporarily):
```python
print("Testing Type A comparison figure...")
annots_list, annots_idx = load_annotation("hico_ground")
results = load_ground_results("hico_ground")
# Use first key in SFT results
first_key = next(iter(results['sft']))
generate_comparison_grounding("hico_ground", first_key, results, annots_idx,
                               OUTPUT_DIR / "hico_ground" / "comparison")
print("Type A grounding test done")
```

```bash
python generate_visualizations.py
ls visualizations/hico_ground/comparison/
```

Verify a PNG file was created.

**Step 4: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add Type A 4-panel comparison figures for grounding and referring"
```

---

## Task 9: Type B — 3-Panel Detail Visualization (Proposals | Prediction | GT)

**Files:**
- Modify: `generate_visualizations.py` — add `generate_detail_figure`

**Step 1: Implement 3-panel detail figure**

```python
def generate_detail_figure(task, key, model_label, result_key,
                            results, annots_idx, output_dir):
    """
    Generate Type B: 3-panel figure showing Proposals | Prediction | GT for one model.
    Works for both grounding and referring tasks.
    """
    is_ground = 'ground' in task

    if is_ground:
        fn, aoid = key
        annot = annots_idx.get(key)
    else:
        annot = annots_idx.get(key)
        fn = annot['file_name'] if annot else ''

    if annot is None:
        print(f"  [SKIP] No annotation for {key}")
        return

    img = load_image(fn, task)
    img_arr = np.array(img)
    W, H = img.size
    image_stem = Path(fn).stem

    # Load proposals
    proposals = load_proposals(image_stem)
    prop_groups = []
    for prop in proposals:
        bb_px = prop['bbox']  # already in pixel coords
        conf = prop.get('confidence', 0)
        label = f"{prop['class_name']} {conf:.2f}"
        prop_groups.append({
            "bboxes": [bb_px],
            "color": COLORS["proposal"],
            "label": label,
            "linestyle": "--",
            "linewidth": 1.5,
        })

    # Get prediction and GT
    entry = results[result_key].get(key, {})

    if is_ground:
        action = annot.get('action', '')
        thinking_map = results.get(f'{result_key}_thinking', {})
        thinking = thinking_map.get((fn, action), '') or entry.get('thinking_content', '')
        gt_pairs_px = extract_gt_pairs_pixel(annot)
        if result_key == 'baseline':
            pred_pairs = parse_ground_answer_baseline(entry.get('generated_text', ''), W, H)
        else:
            pred_pairs = parse_ground_answer_sft(entry.get('answer'), W, H)

        gt_groups = [{"bboxes": [[p, o] for p, o in gt_pairs_px],
                      "p_color": COLORS["gt"], "o_color": COLORS["gt"], "label": "GT pair"}]
        pred_groups = [{"bboxes": [[p, o] for p, o in pred_pairs],
                        "p_color": COLORS["person"], "o_color": COLORS["object"],
                        "label": "Pred pair"}]
        gt_title   = f"Ground Truth\n{len(gt_pairs_px)} pair(s)"
        pred_title = f"{model_label} Prediction\n{len(pred_pairs)} pair(s)"
        fig_title  = f"{fn}  |  {action}  |  {model_label}"
    else:
        thinking = entry.get('thinking_content', '')
        gt_action = annot.get('gt_action', '')
        pred = entry.get('prediction', '—')
        p_bb_px, o_bb_px = extract_refer_input_bboxes_pixel(annot)
        gt_groups  = [{"bboxes": [p_bb_px], "color": COLORS["person"], "label": "Person"},
                      {"bboxes": [o_bb_px], "color": COLORS["object"],  "label": "Object"}]
        pred_groups = gt_groups  # same input bboxes; prediction is text
        gt_title   = f"Ground Truth\n{gt_action}"
        pred_title = f"{model_label}\nPred: {pred}"
        fig_title  = f"{fn}  |  GT: {gt_action}  |  {model_label}"

    # Build figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(fig_title, fontsize=9, y=1.01)

    # Panel 1: Proposals
    draw_bboxes_on_ax(axes[0], img_arr, prop_groups,
                      f"Proposals ({len(proposals)} detected)", fontsize=7)

    # Panel 2: Prediction
    think_caption = wrap_thinking(thinking, max_lines=3)
    pred_full_title = pred_title + f"\n[{think_caption}]" if thinking else pred_title
    draw_bboxes_on_ax(axes[1], img_arr, pred_groups, pred_full_title, fontsize=7)

    # Panel 3: GT
    draw_bboxes_on_ax(axes[2], img_arr, gt_groups, gt_title, fontsize=8)

    plt.tight_layout()

    # Tool call summary as figure annotation
    tool_calls = entry.get('tool_calls', [])
    if tool_calls:
        tool_summary = " → ".join(
            f"{tc['name']}({tc.get('bbox','')})" if tc['name'] == 'zoom_in' else tc['name']
            for tc in tool_calls
        )
        fig.text(0.5, -0.02, f"Tools: {tool_summary[:120]}", ha='center', fontsize=6, color='gray')

    safe_name = Path(fn).stem
    if is_ground:
        safe_name += f"__{key[1].replace('/', '_')}"
    else:
        safe_name += f"__triplet{key}"
    out_path = output_dir / f"{safe_name}_{result_key}_detail.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")
```

**Step 2: Test on one sample**

```python
print("Testing Type B detail figure...")
first_key = next(iter(results['sft']))
generate_detail_figure("hico_ground", first_key, "SFT", "sft",
                        results, annots_idx, OUTPUT_DIR / "hico_ground" / "sft_detail")
print("Type B test done")
```

```bash
python generate_visualizations.py
ls visualizations/hico_ground/sft_detail/
```

**Step 3: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add Type B 3-panel proposals|prediction|GT detail figures"
```

---

## Task 10: Type C — Reasoning Trajectory Text Files

**Files:**
- Modify: `generate_visualizations.py` — add `save_reasoning_trajectory`

**Step 1: Implement reasoning trajectory saver**

```python
def save_reasoning_trajectory(task, key, results, annots_idx, output_dir):
    """
    Save Type C: reasoning trajectory as a structured .txt file.
    Includes: sample metadata, tool call sequence, full thinking for SFT and GRPO.
    """
    is_ground = 'ground' in task

    if is_ground:
        fn, aoid = key
        annot = annots_idx.get(key, {})
        action = annot.get('action', aoid)
        obj_cat = annot.get('object_category', '')
        gt_pairs = extract_gt_pairs_pixel(annot) if annot else []
        context = f"Action: {action}  |  Object: {obj_cat}  |  GT pairs: {len(gt_pairs)}"
    else:
        annot = annots_idx.get(key, {})
        fn = annot.get('file_name', f'triplet_{key}')
        gt_action = annot.get('gt_action', '')
        context = f"GT Action: {gt_action}  |  Triplet ID: {key}"

    lines = [
        "=" * 80,
        f"REASONING TRAJECTORY",
        f"Task:  {task}",
        f"Image: {fn}",
        context,
        "=" * 80,
        "",
    ]

    for model_label, result_key in [("SFT", "sft"), ("SFT-GRPO", "grpo")]:
        entry = results[result_key].get(key, {})
        think_map = results.get(f'{result_key}_thinking', {})

        # Get thinking content
        thinking = entry.get('thinking_content', '')
        if not thinking and is_ground:
            action_str = annot.get('action', '') if annot else ''
            thinking = think_map.get((fn, action_str), '')
        elif not thinking:
            thinking = think_map.get((fn, ''), '')

        # Get tool calls
        tool_calls = entry.get('tool_calls', [])

        # Get prediction
        if is_ground:
            answer = entry.get('answer', 'None')
            n_pred = entry.get('num_pred_pairs', 0)
            pred_summary = f"Predicted {n_pred} pair(s)\nAnswer:\n{answer}"
        else:
            pred = entry.get('prediction', '—')
            exact = entry.get('exact_match', False)
            gt = entry.get('ground_truth', '')
            pred_summary = f"Prediction: {pred}\nGround Truth: {gt}\nExact Match: {exact}"

        lines += [
            f"{'─' * 40}",
            f"MODEL: {model_label}",
            f"{'─' * 40}",
            "",
            "[ TOOL CALL SEQUENCE ]",
        ]
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                name = tc.get('name', '?')
                turn = tc.get('turn', i)
                bbox = tc.get('bbox', '')
                if bbox:
                    lines.append(f"  Turn {turn}: {name}(bbox={bbox})")
                else:
                    lines.append(f"  Turn {turn}: {name}()")
        else:
            lines.append("  (no tool calls)")

        lines += [
            "",
            "[ PREDICTION ]",
            pred_summary,
            "",
            "[ REASONING PROCESS ]",
            thinking if thinking else "(no thinking content available)",
            "",
        ]

    # Save
    safe_name = Path(fn).stem
    if is_ground:
        safe_name += f"__{key[1].replace('/', '_')}"
    else:
        safe_name += f"__triplet{key}"
    out_path = output_dir / f"{safe_name}_reasoning.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {out_path.name}")
```

**Step 2: Test**

```python
save_reasoning_trajectory("hico_ground", first_key, results, annots_idx,
                           OUTPUT_DIR / "hico_ground" / "reasoning")
print("Type C test done")
```

```bash
python generate_visualizations.py
cat visualizations/hico_ground/reasoning/*.txt | head -60
```

**Step 3: Commit**

```bash
git add generate_visualizations.py
git commit -m "feat: add Type C reasoning trajectory text file generator"
```

---

## Task 11: Main Orchestrator — Wire All Tasks Together

**Files:**
- Modify: `generate_visualizations.py` — replace `__main__` with full orchestration

**Step 1: Implement `run_task` function**

```python
def run_task(task):
    """
    Run the full visualization pipeline for one task.
    Loads data → scores → selects → generates all three viz types.
    """
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    is_ground = 'ground' in task

    # Load data
    print("Loading annotations...")
    annots_list, annots_idx = load_annotation(task)

    print("Loading results...")
    if is_ground:
        results = load_ground_results(task)
    else:
        results = load_refer_results(task)

    # Score samples (SFT + GRPO only; baseline gets score=0)
    print("Scoring samples for wrong-proposal detection...")
    scored = score_all_samples(task, results, annots_list, annots_idx)

    # Report wrong-proposal statistics
    total_scored = len(scored)
    wp_cases = [(k, max(v.get('sft',(0,{}))[0], v.get('grpo',(0,{}))[0]))
                for k, v in scored.items()
                if max(v.get('sft',(0,{}))[0], v.get('grpo',(0,{}))[0]) >= WRONG_PROPOSAL_THRESHOLD]
    wp_cases.sort(key=lambda x: -x[1])
    print(f"  Scored {total_scored} samples; {len(wp_cases)} wrong-proposal cases "
          f"(score >= {WRONG_PROPOSAL_THRESHOLD})")

    # Select images
    print("Selecting images...")
    selected_keys = select_images_for_task(
        task, results, annots_list, annots_idx, scored)
    print(f"  Selected {len(selected_keys)} samples")

    # Build and save manifest
    build_selection_manifest(task, selected_keys, scored, results)

    # Generate visualizations
    comp_dir  = OUTPUT_DIR / task / "comparison"
    sft_dir   = OUTPUT_DIR / task / "sft_detail"
    grpo_dir  = OUTPUT_DIR / task / "grpo_detail"
    reason_dir= OUTPUT_DIR / task / "reasoning"

    print("Generating Type A comparison figures...")
    for key in selected_keys:
        try:
            if is_ground:
                generate_comparison_grounding(task, key, results, annots_idx, comp_dir)
            else:
                generate_comparison_referring(task, key, results, annots_idx, comp_dir)
        except Exception as e:
            print(f"  [ERROR] Type A for {key}: {e}")

    print("Generating Type B detail figures (SFT)...")
    for key in selected_keys:
        try:
            generate_detail_figure(task, key, "SFT", "sft",
                                   results, annots_idx, sft_dir)
        except Exception as e:
            print(f"  [ERROR] Type B SFT for {key}: {e}")

    print("Generating Type B detail figures (SFT-GRPO)...")
    for key in selected_keys:
        try:
            generate_detail_figure(task, key, "SFT-GRPO", "grpo",
                                   results, annots_idx, grpo_dir)
        except Exception as e:
            print(f"  [ERROR] Type B GRPO for {key}: {e}")

    print("Generating Type C reasoning trajectories...")
    for key in selected_keys:
        try:
            save_reasoning_trajectory(task, key, results, annots_idx, reason_dir)
        except Exception as e:
            print(f"  [ERROR] Type C for {key}: {e}")

    print(f"Done: {task}")
    return selected_keys, scored
```

**Step 2: Implement `main` with global wrong-proposal count check**

```python
def main():
    setup_output_dirs()
    verify_paths()
    _test_utils()

    all_selected = {}
    all_scored   = {}

    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        selected, scored = run_task(task)
        all_selected[task] = selected
        all_scored[task]   = scored

    # Global wrong-proposal count check (must be >= 5 across all tasks)
    total_wp = 0
    for task, selected in all_selected.items():
        scored = all_scored[task]
        for key in selected:
            max_score = max(scored.get(key, {}).get(m, (0,{}))[0]
                            for m in ('sft', 'grpo'))
            if max_score >= WRONG_PROPOSAL_THRESHOLD:
                total_wp += 1

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for task, selected in all_selected.items():
        print(f"  {task}: {len(selected)} images selected")
    print(f"  Total wrong-proposal cases across all tasks: {total_wp}")
    if total_wp < 5:
        print(f"  WARNING: Only {total_wp} wrong-proposal cases found (need >= 5). "
              f"Consider lowering WRONG_PROPOSAL_THRESHOLD.")
    else:
        print(f"  OK: >= 5 wrong-proposal cases found.")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

**Step 3: Run full pipeline**

```bash
cd /workspace/hoi-benchmarks
source .venv/bin/activate
python generate_visualizations.py 2>&1 | tee visualizations/run.log
```

Expected output per task:
- "Scored N samples; M wrong-proposal cases..."
- "Selected 10 samples"
- Type A, B, C files saved
- Final summary with ≥5 wrong-proposal cases

**Step 4: Verify outputs**

```bash
# Count generated files per task
for task in hico_ground hico_refer swig_ground swig_refer; do
    echo "=== $task ==="
    echo "  comparison: $(ls visualizations/$task/comparison/*.png 2>/dev/null | wc -l)"
    echo "  sft_detail: $(ls visualizations/$task/sft_detail/*.png 2>/dev/null | wc -l)"
    echo "  grpo_detail: $(ls visualizations/$task/grpo_detail/*.png 2>/dev/null | wc -l)"
    echo "  reasoning: $(ls visualizations/$task/reasoning/*.txt 2>/dev/null | wc -l)"
done

# Verify manifest has correct fields
python3 -c "
import json
with open('visualizations/hico_ground/selection_manifest.json') as f:
    m = json.load(f)
print('hico_ground manifest entries:', len(m))
print('wrong-proposal cases:', sum(1 for e in m if max(e['sft_score'], e['grpo_score']) >= 3))
"
```

Expected per task: 10 comparison, 10 sft_detail, 10 grpo_detail, 10 reasoning files.

**Step 5: Commit**

```bash
git add generate_visualizations.py visualizations/
git commit -m "feat: complete visualization pipeline with orchestrator and all 4 tasks"
```

---

## Task 12: Verify Wrong-Proposal Cases and Final Review

**Files:**
- No changes; verification only

**Step 1: Print the wrong-proposal cases per task**

```bash
python3 << 'EOF'
import json
from pathlib import Path

for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
    p = Path(f"visualizations/{task}/selection_manifest.json")
    if not p.exists():
        continue
    with open(p) as f:
        m = json.load(f)
    wp = [e for e in m if max(e['sft_score'], e['grpo_score']) >= 3]
    print(f"\n=== {task}: {len(wp)} wrong-proposal cases ===")
    for e in wp:
        print(f"  {e['file_name']}  SFT={e['sft_score']} GRPO={e['grpo_score']}")
        if e['sft_reasons']:
            print(f"    SFT reasons: {list(e['sft_reasons'].keys())}")
        if e['grpo_reasons']:
            print(f"    GRPO reasons: {list(e['grpo_reasons'].keys())}")
EOF
```

**Step 2: Verify mandatory images appear in all tasks**

```bash
python3 << 'EOF'
import json
from pathlib import Path

mandatory = {
    "hico_ground": ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "hico_refer":  ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "swig_ground": ["checking_24"],
    "swig_refer":  ["checking_24"],
}
all_ok = True
for task, stems in mandatory.items():
    p = Path(f"visualizations/{task}/selection_manifest.json")
    with open(p) as f:
        m = json.load(f)
    fns = [e['file_name'] for e in m]
    for stem in stems:
        found = any(stem in fn for fn in fns)
        status = "OK" if found else "MISSING"
        print(f"  [{status}] {task}: {stem}")
        if not found:
            all_ok = False
print("\nAll mandatory images present:", all_ok)
EOF
```

**Step 3: Spot-check a reasoning trajectory**

```bash
cat visualizations/hico_ground/reasoning/*.txt | head -80
```

Verify it shows: sample metadata, tool call sequence, prediction, and full thinking content for SFT and SFT-GRPO.

**Step 4: Spot-check a comparison figure**

```bash
ls -lh visualizations/hico_ground/comparison/
# Open one PNG to visually verify layout (4 panels, correct bboxes)
```

**Step 5: Final commit**

```bash
git add visualizations/
git commit -m "chore: add generated visualizations and reasoning trajectories for all 4 HOI tasks"
```

---

## Execution Notes

- All Python commands: `source .venv/bin/activate && python generate_visualizations.py`
- If a JSONL thinking file doesn't exist for a model-task pair, the loader returns `{}` gracefully (no crash)
- If a proposal file is missing for an image, `load_proposals()` returns `[]` and Signal 3 is skipped
- The `answer is None` cases (1,404 found in GRPO HICO ground) will render as empty prediction panels — this is intentional and visually informative
- If total wrong-proposal count < 5 after selection, lower `WRONG_PROPOSAL_THRESHOLD` from 3 to 2 and re-run
