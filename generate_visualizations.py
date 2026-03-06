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
REPO = Path(__file__).parent.resolve()
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
        "grpo_think":REPO / "results-sft-grpo-step140/swig_action_sft/swig_action_sft_results_20260226_143917_thinking.jsonl",
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
WRONG_PROPOSAL_THRESHOLD = 3
BBOX_DIVERGE_IOU_THRESH   = 0.15
ZOOM_DIVERGE_IOU_THRESH   = 0.05
IMAGES_PER_TASK           = 10

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "person":   "#FF4444",
    "object":   "#4444FF",
    "gt":       "#22AA22",
    "proposal": "#FF8800",
    "pred":     "#AA22FF",
    "baseline": "#00AAAA",
}


def setup_output_dirs():
    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        for subdir in ["comparison", "sft_detail", "grpo_detail", "reasoning"]:
            (OUTPUT_DIR / task / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Output dirs created under {OUTPUT_DIR}")


def verify_paths():
    missing = []
    for label, d in [("PROPOSALS_DIR", PROPOSALS_DIR), ("HICO_IMAGES", HICO_IMAGES),
                     ("SWIG_IMAGES", SWIG_IMAGES), ("ANNOT_DIR", ANNOT_DIR)]:
        if not d.exists():
            missing.append(f"  [dir][{label}]: {d}")
    for task, paths in RESULTS.items():
        for key, p in paths.items():
            if p is not None and not Path(p).exists():
                missing.append(f"  [{task}][{key}]: {p}")
    for task, p in ANNOT_FILES.items():
        if not Path(p).exists():
            missing.append(f"  [annot][{task}]: {p}")
    if missing:
        detail = "\n".join(missing)
        raise FileNotFoundError(f"Fix missing files before continuing:\n{detail}")
    print("All source files verified OK")


# ── Coordinate utilities ──────────────────────────────────────────────────────

def scale_to_pixel(bbox_1000, width, height):
    """Convert [0,1000]-normalized bbox to pixel coords. Returns [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox_1000
    # x2/y2 clamped to width/height (exclusive-end convention for drawing, not array indexing)
    return [
        max(0, int(x1 * width / 1000)),
        max(0, int(y1 * height / 1000)),
        min(width - 1,  int(x2 * width / 1000)),
        min(height - 1, int(y2 * height / 1000)),
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
            person_entry = next(
                (o for o in objs if isinstance(o, dict) and
                 o.get('label', '').lower() in ('person', 'human', 'man', 'woman', 'player', 'rider', 'athlete')),
                None
            )
            # Fallback: if no person label found, treat first as person, second as object
            if person_entry is None and len(objs) >= 2:
                person_entry, obj_entry = objs[0], objs[1]
            else:
                obj_entry = next(
                    (o for o in objs if isinstance(o, dict) and o is not person_entry
                     and 'bbox_2d' in o),
                    None
                )
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
    text = re.sub(r'```\w*', '', generated_text).strip()
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
    Used for bbox divergence check (Signal 3). No coordinate conversion.
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


# ── GT extractors ─────────────────────────────────────────────────────────────

def extract_gt_pairs_pixel(annot_entry):
    """
    Extract GT (person_bbox, object_bbox) pairs in pixel coords from annotation entry.
    gt_box_inds: flat list [p1_idx, o1_idx, p2_idx, o2_idx, ...]
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
    These are the INPUT bboxes given to the model (in pixel coords).
    """
    boxes = annot_entry['boxes']
    p_bb = boxes[annot_entry['person_box_idx']]
    o_bb = boxes[annot_entry['object_box_idx']]
    return p_bb, o_bb


# ── Smoke tests ───────────────────────────────────────────────────────────────

def _test_utils():
    # Coordinate round-trip
    assert scale_to_pixel([500, 500, 750, 750], 640, 480) == [320, 240, 480, 360]
    assert scale_to_1000([320, 240, 480, 360], 640, 480) == [500, 500, 750, 750]
    # IoU edge cases
    a = [0, 0, 100, 100]
    assert bbox_iou(a, a) == 1.0
    assert bbox_iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    # Partial overlap: two 100x100 boxes sharing 50x50
    expected = 2500 / (20000 - 2500)
    assert abs(bbox_iou([0, 0, 100, 100], [50, 50, 150, 150]) - expected) < 1e-6
    # SFT answer parsing
    sft_ans = '[{"bbox_2d": [100, 200, 300, 400], "label": "person"}, {"bbox_2d": [500, 600, 700, 800], "label": "bicycle"}]'
    pairs = parse_ground_answer_sft(sft_ans, 1000, 1000)
    assert len(pairs) == 1
    assert pairs[0][0] == [100, 200, 300, 400]
    # Baseline answer parsing
    baseline_ans = '[{"pair_id": 1, "person_bbox": [100, 200, 300, 400], "object_bbox": [500, 600, 700, 800]}]'
    pairs = parse_ground_answer_baseline(baseline_ans, 1000, 1000)
    assert len(pairs) == 1
    # None/empty inputs return []
    assert parse_ground_answer_sft(None, 640, 480) == []
    assert parse_ground_answer_baseline(None, 640, 480) == []
    assert extract_all_pred_bboxes_1000(None) == []
    # GT extractor
    fake_annot = {"boxes": [[10,10,50,50],[60,60,100,100],[20,20,40,40],[70,70,90,90]],
                  "gt_box_inds": [0, 1, 2, 3]}
    pairs = extract_gt_pairs_pixel(fake_annot)
    assert len(pairs) == 2
    assert pairs[0] == ([10,10,50,50], [60,60,100,100])
    # Refer extractor
    fake_refer = {"boxes": [[10,10,50,50],[60,60,100,100]], "person_box_idx": 0, "object_box_idx": 1}
    p, o = extract_refer_input_bboxes_pixel(fake_refer)
    assert p == [10,10,50,50]
    assert o == [60,60,100,100]
    print("_test_utils: all assertions passed")


if __name__ == "__main__":
    setup_output_dirs()
    verify_paths()
    _test_utils()
    print("Setup complete.")
