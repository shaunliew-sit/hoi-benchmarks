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


# ── Data loaders ───────────────────────────────────────────────────────────────

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
    Baseline entry has keys: file_name, action, object, action_object_id, prompt, thinking_content, generated_text, matches_per_threshold
    SFT/GRPO entry has keys: file_name, action, object, num_gt_pairs, num_pred_pairs, tool_calls, answer, matches_per_threshold
    """
    paths = RESULTS[task]

    def _index(json_path):
        idx = {}
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            fn = entry.get('file_name', '')
            aoid = entry.get('action_object_id',
                             f"{entry.get('action', '')}_{entry.get('object_category', entry.get('object', ''))}")
            idx[(fn, aoid)] = entry
        return idx

    baseline_idx = _index(paths['baseline'])
    sft_idx      = _index(paths['sft'])
    grpo_idx     = _index(paths['grpo'])

    return {
        "baseline": baseline_idx,
        "sft":      sft_idx,
        "grpo":     grpo_idx,
        "sft_thinking":  load_thinking_jsonl(paths.get('sft_think')),
        "grpo_thinking": load_thinking_jsonl(paths.get('grpo_think')),
    }


def load_refer_results(task):
    """
    Load action referring results for all 3 models for a given task (hico_refer or swig_refer).
    Returns dict:
      {
        "baseline": {triplet_id: entry, ...},
        "sft":      {triplet_id: entry, ...},
        "grpo":     {triplet_id: entry, ...},
        "sft_thinking":  {(file_name, action): thinking_str},
        "grpo_thinking": {(file_name, action): thinking_str},
      }
    Entry keys: triplet_id, file_name, ground_truth, prediction, tool_calls, exact_match, thinking_content
    """
    paths = RESULTS[task]

    def _index(json_path):
        idx = {}
        fallback_counter = 0
        with open(json_path) as f:
            data = json.load(f)
        for entry in data:
            if 'triplet_id' in entry:
                tid = entry['triplet_id']
            else:
                tid = fallback_counter
                fallback_counter += 1
            idx[tid] = entry
        return idx

    baseline_idx = _index(paths['baseline'])
    sft_idx      = _index(paths['sft'])
    grpo_idx     = _index(paths['grpo'])

    return {
        "baseline": baseline_idx,
        "sft":      sft_idx,
        "grpo":     grpo_idx,
        "sft_thinking":  load_thinking_jsonl(paths.get('sft_think')),
        "grpo_thinking": load_thinking_jsonl(paths.get('grpo_think')),
    }


def load_annotation(task):
    """
    Load annotation for a task.
    Grounding: returns (list, dict) where dict is keyed by (file_name, action_object_id).
    Referring: returns (list, dict) where dict is keyed by triplet_id (positional index).
    """
    with open(ANNOT_FILES[task]) as f:
        annots = json.load(f)

    if 'ground' in task:
        idx = {}
        for entry in annots:
            fn = entry['file_name']
            aoid = entry.get('action_object_id',
                             f"{entry.get('action', '')}_{entry.get('object_category', '')}")
            idx[(fn, aoid)] = entry
        return annots, idx
    else:
        idx = {i: entry for i, entry in enumerate(annots)}
        return annots, idx


# ── Wrong-proposal detection ───────────────────────────────────────────────────

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


def score_text_signals(thinking_content):
    """
    Score a thinking string for Signals 2 (high-precision) and 5 (medium) keywords.
    Returns (signal2_score, signal5_score, matched_keywords).
    matched_keywords is list of ("HIGH"|"MED", keyword_string) tuples.
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
    Signal 1: empty/None response (grounding only).
    Returns (score, list_of_reasons).
    """
    if not is_ground:
        return 0, []
    reasons = []
    if entry.get('answer') is None:
        reasons.append("answer=None")
    if entry.get('num_pred_pairs', -1) == 0:
        reasons.append("num_pred_pairs=0")
    if isinstance(entry.get('answer'), str):
        stripped = entry['answer'].strip()
        if stripped == '':
            reasons.append("answer=empty_string")
        elif stripped == '[]':
            reasons.append("answer='[]'")
    return (5 if reasons else 0), reasons


def signal3_bbox_diverge(entry, proposals):
    """
    Signal 3 (grounding only): any predicted bbox has max IoU < BBOX_DIVERGE_IOU_THRESH
    with ALL proposal bboxes_1000.
    Returns (score, list_of_diverged_info) or (0, []).
    """
    if proposals is None or len(proposals) == 0:
        return 0, []

    answer = entry.get('answer')
    if not answer:
        return 0, []

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


def signal4_zoom_diverge(entry, annot_entry):
    """
    Signal 4 (action referring only): ALL zoom_in tool calls target a region
    with IoU < ZOOM_DIVERGE_IOU_THRESH with BOTH the given person AND object boxes.
    Returns (score, list_of_diverged_zoom_info) or (0, []).
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
        diverged_zooms.append({
            'zoom_bbox': zoom_bb,
            'person_iou': round(p_iou, 3),
            'object_iou': round(o_iou, 3),
        })

    if all_diverged and diverged_zooms:
        return 3, diverged_zooms
    return 0, []


def score_sample(entry, thinking_content, is_ground,
                 annot_entry=None, proposals=None, model=None):
    """
    Compute wrong-proposal score for a single sample.
    Returns (total_score, reasons_dict).

    Signals:
      1 (+5): empty/None answer (grounding only)
      2 (+4 each): high-precision keywords in thinking
      3 (+3): predicted bbox IoU < threshold vs all proposals (grounding)
      4 (+3): all zoom_in calls miss given person+object boxes (referring)
      5 (+2 each): medium-confidence keywords in thinking
    """
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


# ── Sample scoring + image selection ──────────────────────────────────────────

def score_all_samples(task, results, annots_list, annots_idx):
    """
    Score all SFT and GRPO samples using the 5-signal wrong-proposal detection.
    Returns dict: {lookup_key: {"sft": (score, reasons), "grpo": (score, reasons)}}
    where lookup_key is (file_name, action_object_id) for grounding, triplet_id for referring.
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
                tid = entry.get('triplet_id')
                annot = annots_idx.get(tid) if tid is not None else None

            # Get thinking content (prefer inline field, fall back to JSONL map)
            if is_ground:
                action = entry.get('action', '')
                thinking = entry.get('thinking_content', '') or think_map.get((fn, action), '')
            else:
                thinking = entry.get('thinking_content', '')
                if not thinking:
                    thinking = think_map.get((fn, ''), '')

            # Load proposals (grounding needs them for Signal 3)
            proposals = load_proposals(image_stem) if is_ground else None

            score, reasons = score_sample(
                entry=entry,
                thinking_content=thinking,
                is_ground=is_ground,
                annot_entry=annot,
                proposals=proposals,
                model=model,
            )

            if lookup_key not in scored:
                scored[lookup_key] = {}
            scored[lookup_key][model] = (score, reasons)

    return scored


def select_images_for_task(task, results, annots_list, annots_idx, scored,
                            n=IMAGES_PER_TASK):
    """
    Select n samples for visualization.
    Priority:
      1. Mandatory images (MUST include — one entry per mandatory image stem)
      2. Wrong-proposal cases (score >= WRONG_PROPOSAL_THRESHOLD), highest score first
      3. Fill remaining slots with diverse images (one per unique image stem)

    Returns list of lookup_keys.
    """
    is_ground = 'ground' in task
    mandatory_stems = MANDATORY.get(task, [])
    selected = []
    selected_set = set()
    selected_stems = set()

    def get_stem(key):
        if is_ground:
            fn = key[0]
        else:
            entry = results['sft'].get(key) or results['baseline'].get(key) or {}
            fn = entry.get('file_name', '')
        return Path(fn).stem

    # Pass 1: mandatory images
    all_keys = list(results['sft'].keys())
    for stem in mandatory_stems:
        for key in all_keys:
            if get_stem(key) == stem and key not in selected_set:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)
                break  # one entry per mandatory image

    # Pass 2: wrong-proposal cases
    wrong_cases = []
    for key, model_scores in scored.items():
        max_score = max((v[0] for v in model_scores.values()), default=0)
        if max_score >= WRONG_PROPOSAL_THRESHOLD and key not in selected_set:
            stem = get_stem(key)
            if stem not in selected_stems:
                wrong_cases.append((max_score, key))
    wrong_cases.sort(key=lambda x: -x[0])
    for score_val, key in wrong_cases:
        if len(selected) >= n:
            break
        selected.append(key)
        selected_set.add(key)
        selected_stems.add(get_stem(key))

    # Pass 3: fill with diverse samples
    if len(selected) < n:
        import random
        rng = random.Random(42)
        shuffled_keys = list(all_keys)
        rng.shuffle(shuffled_keys)
        for key in shuffled_keys:
            if len(selected) >= n:
                break
            stem = get_stem(key)
            if key not in selected_set and stem not in selected_stems:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)

    return selected[:n]


def build_selection_manifest(task, selected_keys, scored, results):
    """Build and save selection_manifest.json for a task. Returns the manifest list."""
    is_ground = 'ground' in task
    manifest = []
    for key in selected_keys:
        if is_ground:
            entry = results['sft'].get(key) or results['baseline'].get(key) or {}
            fn = entry.get('file_name', '')
            action = entry.get('action', '')
        else:
            entry = results['sft'].get(key) or results['baseline'].get(key) or {}
            fn = entry.get('file_name', '')
            action = entry.get('ground_truth', '')

        model_scores = scored.get(key, {})
        manifest.append({
            "key": str(key),
            "file_name": fn,
            "action": action,
            "sft_score":   model_scores.get('sft',  (0, {}))[0],
            "grpo_score":  model_scores.get('grpo', (0, {}))[0],
            "sft_reasons": model_scores.get('sft',  (0, {}))[1],
            "grpo_reasons":model_scores.get('grpo', (0, {}))[1],
        })
    out_path = OUTPUT_DIR / task / "selection_manifest.json"
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {out_path.name}")
    return manifest


if __name__ == "__main__":
    setup_output_dirs()
    verify_paths()
    _test_utils()

    print("Testing image selection on hico_ground...")
    annots_list, annots_idx = load_annotation("hico_ground")
    results = load_ground_results("hico_ground")
    scored = score_all_samples("hico_ground", results, annots_list, annots_idx)
    print(f"  Scored {len(scored)} samples")

    # Count wrong-proposal cases
    wp_cases = [(k, max(v.get('sft',(0,{}))[0], v.get('grpo',(0,{}))[0]))
                for k, v in scored.items()
                if max(v.get('sft',(0,{}))[0], v.get('grpo',(0,{}))[0]) >= WRONG_PROPOSAL_THRESHOLD]
    print(f"  Wrong-proposal cases (score >= {WRONG_PROPOSAL_THRESHOLD}): {len(wp_cases)}")

    selected = select_images_for_task("hico_ground", results, annots_list, annots_idx, scored)
    print(f"  Selected {len(selected)} samples")

    # Verify mandatory images are present
    for stem in MANDATORY["hico_ground"]:
        found = any(stem in str(k) for k in selected)
        assert found, f"Mandatory image {stem} not in selection!"
    print(f"  All mandatory images present: {MANDATORY['hico_ground']}")

    # Count wrong-proposal cases in selection
    wp_in_sel = sum(
        1 for k in selected
        if max(scored.get(k, {}).get(m, (0, {}))[0] for m in ('sft', 'grpo')) >= WRONG_PROPOSAL_THRESHOLD
    )
    print(f"  Wrong-proposal cases in selection: {wp_in_sel}")

    manifest = build_selection_manifest("hico_ground", selected, scored, results)
    assert len(manifest) == len(selected)
    print(f"  Manifest saved with {len(manifest)} entries")

    print("Task 6 tests passed")
    print("Setup complete.")
