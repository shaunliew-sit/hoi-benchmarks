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
        "grpo":     REPO / "results-sft-grpo-new/hico_ground_sft_new/hico_ground_sft_results_20260311_025017.json",
        "grpo_think":REPO / "results-sft-grpo-new/hico_ground_sft_new/hico_ground_sft_results_20260311_025017_thinking.jsonl",
    },
    "hico_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/hico_action_qwen3vl_instruct/hico_action_qwen3vl_results_20260302_132528_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-new/hico_action_sft_new/hico_action_sft_results_20260310_141500_per_triplet.json",
        "grpo_think":REPO / "results-sft-grpo-new/hico_action_sft_new/hico_action_sft_results_20260310_141500_thinking.jsonl",
    },
    "swig_ground": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_ground_qwen3vl_instruct/swig_ground_qwen3vl_results_20260302_132422.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-new/swig_ground_sft_new/swig_ground_sft_results_20260311_092108.json",
        "grpo_think":REPO / "results-sft-grpo-new/swig_ground_sft_new/swig_ground_sft_results_20260311_092108_thinking.jsonl",
    },
    "swig_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_action_qwen3vl_instruct/swig_action_qwen3vl_results_20260302_132455_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-new/swig_action_sft_new/swig_action_sft_results_20260310_141513_per_triplet.json",
        "grpo_think":REPO / "results-sft-grpo-new/swig_action_sft_new/swig_action_sft_results_20260310_141513_thinking.jsonl",
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
    "hico_ground": ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612", "HICO_test2015_00006692"],
    "hico_refer":  ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612", "HICO_test2015_00006692"],
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
    "gt_alt":   "#FF6B00",   # orange — used when background is green
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
            # Always build key from action+object to match annotation index and SFT/GRPO.
            # Baseline has action_object_id="1_87_bench" (different format) so we ignore it.
            aoid = f"{entry.get('action', '')}_{entry.get('object_category', entry.get('object', ''))}"
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
            # Build key to match results indexing: "{action}_{object_category}"
            aoid = f"{entry.get('action', '')}_{entry.get('object_category', '')}"
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


def _get_action_for_key(key, task, results, annots_idx):
    """Return the action label for a lookup key (for diversity bucketing)."""
    is_ground = 'ground' in task
    if is_ground:
        annot = annots_idx.get(key)
        return annot.get('action', '') if annot else ''
    else:
        annot = annots_idx.get(key)
        return annot.get('gt_action', '') if annot else ''


def select_images_for_task(task, results, annots_list, annots_idx, scored,
                            n=IMAGES_PER_TASK):
    """
    Select n samples for visualization.
    Priority:
      1. Mandatory images (MUST include — one entry per mandatory image stem)
      2. Wrong-proposal cases (score >= WRONG_PROPOSAL_THRESHOLD), highest score first,
         capped at 1 image per action category for diversity
      3. Fill remaining slots with diverse images (one per unique image stem,
         one per action category)

    Returns list of lookup_keys.
    """
    is_ground = 'ground' in task
    mandatory_stems = MANDATORY.get(task, [])
    selected = []
    selected_set = set()
    selected_stems = set()
    selected_actions = set()

    def get_stem(key):
        if is_ground:
            fn = key[0]
        else:
            entry = results['sft'].get(key) or results['baseline'].get(key) or {}
            fn = entry.get('file_name', '')
        return Path(fn).stem

    def get_action(key):
        return _get_action_for_key(key, task, results, annots_idx)

    # Pass 1: mandatory images (action diversity not enforced here)
    all_keys = list(results['sft'].keys())
    for stem in mandatory_stems:
        for key in all_keys:
            if get_stem(key) == stem and key not in selected_set:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)
                selected_actions.add(get_action(key))
                break  # one entry per mandatory image

    # Pass 2: wrong-proposal cases, diverse actions, highest score first
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
        stem = get_stem(key)
        action = get_action(key)
        if key not in selected_set and stem not in selected_stems and action not in selected_actions:
            selected.append(key)
            selected_set.add(key)
            selected_stems.add(stem)
            selected_actions.add(action)

    # Pass 2b: relax action constraint if still not enough (allow repeated actions)
    for score_val, key in wrong_cases:
        if len(selected) >= n:
            break
        stem = get_stem(key)
        if key not in selected_set and stem not in selected_stems:
            selected.append(key)
            selected_set.add(key)
            selected_stems.add(stem)
            selected_actions.add(get_action(key))

    # Pass 3: fill with diverse samples (action diversity first, then any)
    if len(selected) < n:
        import random
        rng = random.Random(42)
        shuffled_keys = list(all_keys)
        rng.shuffle(shuffled_keys)
        # First pass: prefer unseen actions
        for key in shuffled_keys:
            if len(selected) >= n:
                break
            stem = get_stem(key)
            action = get_action(key)
            if key not in selected_set and stem not in selected_stems and action not in selected_actions:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)
                selected_actions.add(action)
        # Second pass: fill any remaining slots with any unseen stems
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
            "key": list(key) if isinstance(key, tuple) else key,
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


# ── Drawing utilities ──────────────────────────────────────────────────────────

def get_gt_color(img_array):
    """
    Return appropriate GT bbox color based on the image's border region hue.
    Returns the alternate orange color when the background is dominated by green
    (e.g. a sports field), so the GT box remains clearly visible.
    """
    h, w = img_array.shape[:2]
    border_px = max(1, min(15, h // 10, w // 10))
    sample = np.concatenate([
        img_array[:border_px, :, :3].reshape(-1, 3),
        img_array[-border_px:, :, :3].reshape(-1, 3),
        img_array[:, :border_px, :3].reshape(-1, 3),
        img_array[:, -border_px:, :3].reshape(-1, 3),
    ]).astype(float)
    r_mean, g_mean, b_mean = sample[:, 0].mean(), sample[:, 1].mean(), sample[:, 2].mean()
    if g_mean > r_mean * 1.15 and g_mean > b_mean * 1.15 and g_mean > 60:
        return COLORS["gt_alt"]
    return COLORS["gt"]


def load_image(file_name, task):
    """Load PIL image for a given file_name and task (determines image dir)."""
    if 'hico' in task:
        img_path = HICO_IMAGES / file_name
    else:
        img_path = SWIG_IMAGES / file_name
    if not img_path.exists():
        for ext in ('.jpg', '.jpeg', '.png'):
            alt = img_path.with_suffix(ext)
            if alt.exists():
                return Image.open(alt).convert('RGB')
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert('RGB')


def draw_bboxes_on_ax(ax, img_array, bbox_groups, title, title_color='black', fontsize=13):
    """
    Draw an image with colored bounding boxes on a matplotlib Axes.

    bbox_groups: list of dicts, each with one of two formats:

    Pair format (person-object pairs):
      {
        'bboxes': [[p_bbox, o_bbox], ...],  # pixel coords [x1,y1,x2,y2]
        'p_color': hex_str,
        'o_color': hex_str,
        'label': str,
        'linestyle': '-' or '--',
        'linewidth': float,
      }

    Single format (individual boxes):
      {
        'bboxes': [[x1,y1,x2,y2], ...],
        'color': hex_str,
        'label': str,
        'linestyle': '-' or '--',
        'linewidth': float,
      }

    A group is in pair format if its first bbox item is itself a list of lists,
    i.e. isinstance(item[0], list) where item = bboxes[0].
    """
    ax.imshow(img_array)
    ax.set_title(title, color=title_color, fontsize=fontsize,
                 fontweight='bold', pad=4, wrap=True)
    ax.axis('off')

    legend_patches = []

    for group in bbox_groups:
        bboxes   = group.get('bboxes', [])
        p_color  = group.get('p_color', COLORS['pred'])
        o_color  = group.get('o_color', COLORS['pred'])
        s_color  = group.get('color', p_color)
        label    = group.get('label', '')
        ls       = group.get('linestyle', '-')
        lw       = group.get('linewidth', 2.0)

        if not bboxes:
            continue

        # Determine format: pair if first element's first element is a list
        first = bboxes[0]
        is_pair = isinstance(first[0], (list, tuple))

        for item in bboxes:
            if is_pair:
                if len(item) < 2:
                    continue
                p_bb, o_bb = item[0], item[1]
                for bb, color in [(p_bb, p_color), (o_bb, o_color)]:
                    x1, y1, x2, y2 = bb
                    rect = patches.Rectangle(
                        (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                        linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls
                    )
                    ax.add_patch(rect)
                # Connecting line between centers
                p_cx = (p_bb[0] + p_bb[2]) / 2
                p_cy = (p_bb[1] + p_bb[3]) / 2
                o_cx = (o_bb[0] + o_bb[2]) / 2
                o_cy = (o_bb[1] + o_bb[3]) / 2
                ax.plot([p_cx, o_cx], [p_cy, o_cy],
                        color=p_color, lw=1, ls='--', alpha=0.6)
            else:
                x1, y1, x2, y2 = item
                rect = patches.Rectangle(
                    (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                    linewidth=lw, edgecolor=s_color, facecolor='none', linestyle=ls
                )
                ax.add_patch(rect)

        legend_color = s_color if not is_pair else p_color
        if label:
            legend_patches.append(patches.Patch(color=legend_color, label=label))

    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=10,
                  framealpha=0.7, markerscale=0.8)


def wrap_thinking(text, max_lines=4, line_width=60):
    """Truncate and wrap thinking content for display as a panel caption."""
    if not text:
        return "(no thinking)"
    first_para = text.split('\n\n')[0].replace('\n', ' ').strip()
    wrapped = textwrap.fill(first_para, width=line_width)
    lines = wrapped.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(0, line_width - 3)] + '...'
    return '\n'.join(lines)


# ── Type A: 4-panel comparison ─────────────────────────────────────────────────

def generate_comparison_grounding(task, key, results, annots_idx, output_dir):
    """Type A: 4-panel comparison (GT | Baseline | SFT | SFT-GRPO) for grounding."""
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

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{fn}  —  {action} | {obj_cat}", fontsize=13, y=1.01)

    # GT panel
    gt_color = get_gt_color(img_arr)
    gt_groups = [{"bboxes": [[p, o] for p, o in gt_pairs_px],
                  "p_color": gt_color, "o_color": gt_color,
                  "label": "GT pair", "linestyle": "-"}]
    draw_bboxes_on_ax(axes[0], img_arr, gt_groups,
                      f"Ground Truth\n{len(gt_pairs_px)} pair(s)", fontsize=12)

    # Model panels
    for ax, (model_label, result_key) in zip(axes[1:], [
        ("Baseline", "baseline"),
        ("SFT",      "sft"),
        ("SFT-GRPO", "grpo"),
    ]):
        entry = results[result_key].get(key, {})
        thinking_map = results.get(f'{result_key}_thinking', {})
        thinking = entry.get('thinking_content', '') or thinking_map.get((fn, action), '')

        if result_key == 'baseline':
            pred_pairs = parse_ground_answer_baseline(entry.get('generated_text', ''), W, H)
        else:
            pred_pairs = parse_ground_answer_sft(entry.get('answer'), W, H)

        matched = entry.get('matches_per_threshold', {}).get('0.5', {}).get('matched', '?')
        n_tools = len(entry.get('tool_calls', []))
        subtitle = f"{model_label}\npred={len(pred_pairs)}, gt={len(gt_pairs_px)}, matched@0.5={matched}"
        if result_key == 'baseline':
            subtitle += "\n[No Thinking]"
        else:
            subtitle += f"  [tools: {n_tools}]"
            if thinking:
                subtitle += f"\n{wrap_thinking(thinking, max_lines=2, line_width=50)}"

        pred_groups = [{"bboxes": [[p, o] for p, o in pred_pairs],
                        "p_color": COLORS["pred"], "o_color": COLORS["pred"],
                        "label": "Pred pair"}]
        draw_bboxes_on_ax(ax, img_arr, pred_groups, subtitle, fontsize=11)

    plt.tight_layout()
    safe_name = Path(fn).stem + f"__{aoid.replace('/', '_')}"
    out_path = output_dir / f"{safe_name}_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


def generate_comparison_referring(task, key, results, annots_idx, output_dir):
    """Type A: 4-panel comparison (GT | Baseline | SFT | SFT-GRPO) for action referring."""
    annot = annots_idx.get(key)
    if annot is None:
        print(f"  [SKIP] No annotation for triplet_id={key}")
        return

    fn = annot['file_name']
    img = load_image(fn, task)
    img_arr = np.array(img)
    gt_action = annot.get('gt_action', '')

    p_bb_px, o_bb_px = extract_refer_input_bboxes_pixel(annot)
    input_groups = [
        {"bboxes": [p_bb_px], "color": COLORS["person"], "label": "Person", "linewidth": 2.5},
        {"bboxes": [o_bb_px], "color": COLORS["object"],  "label": "Object",  "linewidth": 2.5},
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(f"{fn}  —  GT: {gt_action}", fontsize=13, y=1.01)

    # GT panel (show input bboxes + GT action label)
    draw_bboxes_on_ax(axes[0], img_arr, input_groups,
                      f"Ground Truth\n{gt_action}", fontsize=12)

    # Model panels
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

        t_color = '#22AA22' if exact else '#CC2222'
        subtitle = f"{model_label}\nPred: {pred}"
        if model_label == "Baseline":
            subtitle += "\n[No Thinking]"
        else:
            subtitle += f"  [tools: {n_tools}]"
            if thinking:
                subtitle += f"\n{wrap_thinking(thinking, max_lines=2, line_width=50)}"

        draw_bboxes_on_ax(ax, img_arr, input_groups, subtitle,
                          title_color=t_color, fontsize=11)

    plt.tight_layout()
    stem = Path(fn).stem
    out_path = output_dir / f"{stem}__triplet{key}_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


# ── Type B: 3-panel proposals|prediction|GT ────────────────────────────────────

def get_zoom_crops(img, tool_calls):
    """
    Re-generate zoom crop images from bbox coords stored in tool_calls.
    Uses the same formula as the evaluation script: bbox in [0,1000] → pixel crop.
    Returns list of (turn, bbox_1000, PIL.Image) for each zoom_in call.
    """
    W, H = img.size
    crops = []
    for tc in tool_calls:
        if tc.get('name') != 'zoom_in':
            continue
        bbox = tc.get('bbox', [])
        if len(bbox) != 4:
            continue
        x1 = max(0, int(bbox[0] * W / 1000))
        y1 = max(0, int(bbox[1] * H / 1000))
        x2 = min(W, int(bbox[2] * W / 1000))
        y2 = min(H, int(bbox[3] * H / 1000))
        if x2 <= x1 or y2 <= y1:
            continue
        crops.append((tc.get('turn', len(crops)), bbox, img.crop((x1, y1, x2, y2))))
    return crops


def generate_detail_figure(task, key, model_label, result_key,
                            results, annots_idx, output_dir):
    """Type B: 3-panel detail figure (Proposals | Prediction | GT) for SFT or GRPO."""
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

    # Load proposals — consolidated into one group so the legend stays compact
    proposals = load_proposals(image_stem)
    if proposals:
        prop_groups = [{
            "bboxes": [prop['bbox'] for prop in proposals],
            "color": COLORS["proposal"],
            "label": f"Proposals ({len(proposals)})",
            "linestyle": "--",
            "linewidth": 1.0,
        }]
    else:
        prop_groups = []

    entry = results[result_key].get(key, {})

    if is_ground:
        action = annot.get('action', '')
        thinking_map = results.get(f'{result_key}_thinking', {})
        thinking = entry.get('thinking_content', '') or thinking_map.get((fn, action), '')
        gt_pairs_px = extract_gt_pairs_pixel(annot)
        if result_key == 'baseline':
            pred_pairs = parse_ground_answer_baseline(entry.get('generated_text', ''), W, H)
        else:
            pred_pairs = parse_ground_answer_sft(entry.get('answer'), W, H)

        gt_color = get_gt_color(img_arr)
        gt_groups   = [{"bboxes": [[p, o] for p, o in gt_pairs_px],
                        "p_color": gt_color, "o_color": gt_color, "label": "GT pair"}]
        pred_groups = [{"bboxes": [[p, o] for p, o in pred_pairs],
                        "p_color": COLORS["person"], "o_color": COLORS["object"],
                        "label": "Pred pair"}]
        gt_title    = f"Ground Truth\n{len(gt_pairs_px)} pair(s)"
        pred_title  = f"{model_label}\n{len(pred_pairs)} pair(s) predicted"
        fig_suptitle = f"{fn}  |  {action}  |  {model_label}"
    else:
        thinking = entry.get('thinking_content', '')
        gt_action = annot.get('gt_action', '')
        pred = entry.get('prediction', '—')
        p_bb_px, o_bb_px = extract_refer_input_bboxes_pixel(annot)
        box_groups = [{"bboxes": [p_bb_px], "color": COLORS["person"], "label": "Person"},
                      {"bboxes": [o_bb_px], "color": COLORS["object"],  "label": "Object"}]
        gt_groups   = box_groups
        pred_groups = box_groups
        gt_title    = f"Ground Truth\n{gt_action}"
        pred_title  = f"{model_label}\nPred: {pred}"
        fig_suptitle = f"{fn}  |  GT: {gt_action}  |  {model_label}"

    # Compute zoom crops for SFT/GRPO (baseline has no tool_calls)
    tool_calls = entry.get('tool_calls', [])
    zoom_crops = get_zoom_crops(img, tool_calls) if result_key != 'baseline' else []
    n_crops = len(zoom_crops)

    # Build layout: row 0 = Proposals|Prediction|GT, row 1 = zoom crops (if any)
    n_cols = max(3, n_crops)
    if n_crops > 0:
        fig = plt.figure(figsize=(6 * n_cols, 12), constrained_layout=True)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.45, wspace=0.3)
        ax_prop = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        ax_gt   = fig.add_subplot(gs[0, 2])
        crop_axes = [fig.add_subplot(gs[1, i]) for i in range(n_crops)]
    else:
        fig, (ax_prop, ax_pred, ax_gt) = plt.subplots(1, 3, figsize=(18, 6))
        crop_axes = []

    fig.suptitle(fig_suptitle, fontsize=13, y=1.01)

    # Panel 1: Proposals — use smaller fontsize so boxes don't crowd the panel
    draw_bboxes_on_ax(ax_prop, img_arr, prop_groups,
                      f"Proposals ({len(proposals)} detected)", fontsize=10)

    # Panel 2: Prediction + thinking caption
    think_caption = wrap_thinking(thinking, max_lines=3, line_width=55) if thinking else ''
    pred_full = pred_title + (f"\n{think_caption}" if think_caption else "")
    draw_bboxes_on_ax(ax_pred, img_arr, pred_groups, pred_full, fontsize=11)

    # Panel 3: GT
    draw_bboxes_on_ax(ax_gt, img_arr, gt_groups, gt_title, fontsize=12)

    # Row 2: zoom crops
    for ax, (turn, bbox, crop_img) in zip(crop_axes, zoom_crops):
        ax.imshow(np.array(crop_img))
        ax.set_title(f"Turn {turn}: zoom_in\n{bbox}", fontsize=10)
        ax.axis('off')

    # Tool call sequence as figure footer
    if tool_calls:
        tool_summary = " -> ".join(
            f"zoom_in({tc.get('bbox','')})" if tc.get('name') == 'zoom_in' else tc.get('name','?')
            for tc in tool_calls
        )
        fig.text(0.5, -0.02, f"Tools: {tool_summary[:160]}", ha='center', fontsize=9, color='gray')

    if n_crops == 0:
        plt.tight_layout()

    safe_name = Path(fn).stem
    if is_ground:
        safe_name += f"__{key[1].replace('/', '_')}"
    else:
        safe_name += f"__triplet{key}"
    out_path = output_dir / f"{safe_name}_{result_key}_detail.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path.name}")


# ── Type C: reasoning trajectory text files ────────────────────────────────────

def save_reasoning_trajectory(task, key, results, annots_idx, output_dir):
    """
    Save Type C: reasoning trajectory as a structured .txt file with:
    - Sample metadata
    - Tool call sequence (per model)
    - Full prediction
    - Full thinking content (SFT and GRPO)
    """
    is_ground = 'ground' in task

    if is_ground:
        fn, aoid = key
        annot = annots_idx.get(key)
        action = annot.get('action', aoid) if annot else aoid
        obj_cat = annot.get('object_category', '') if annot else ''
        gt_pairs = extract_gt_pairs_pixel(annot) if annot else []
        context = f"Action: {action}  |  Object: {obj_cat}  |  GT pairs: {len(gt_pairs)}"
    else:
        annot = annots_idx.get(key)
        fn = annot.get('file_name', f'triplet_{key}') if annot else f'triplet_{key}'
        gt_action = annot.get('gt_action', '')
        context = f"GT Action: {gt_action}  |  Triplet ID: {key}"

    lines = [
        "=" * 80,
        "REASONING TRAJECTORY",
        f"Task:  {task}",
        f"Image: {fn}",
        context,
        "=" * 80,
        "",
    ]

    for model_label, result_key in [("SFT", "sft"), ("SFT-GRPO", "grpo")]:
        entry = results[result_key].get(key, {})
        think_map = results.get(f'{result_key}_thinking', {})

        thinking = entry.get('thinking_content', '')
        if not thinking and is_ground:
            action_str = annot.get('action', '') if annot else ''
            thinking = think_map.get((fn, action_str), '')
        elif not thinking:
            thinking = think_map.get((fn, ''), '')

        tool_calls = entry.get('tool_calls', [])

        if is_ground:
            answer = entry.get('answer') or 'None'
            n_pred = entry.get('num_pred_pairs', 0)
            pred_summary = f"Predicted {n_pred} pair(s)\nAnswer:\n{answer}"
        else:
            pred = entry.get('prediction', '—')
            exact = entry.get('exact_match', False)
            gt = entry.get('ground_truth', '')
            pred_summary = f"Prediction:    {pred}\nGround Truth:  {gt}\nExact Match:   {exact}"

        lines += [
            "-" * 40,
            f"MODEL: {model_label}",
            "-" * 40,
            "",
            "[ TOOL CALL SEQUENCE ]",
        ]

        if tool_calls:
            for tc in tool_calls:
                name = tc.get('name', '?')
                turn = tc.get('turn', '?')
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

    safe_name = Path(fn).stem
    if is_ground:
        safe_name += f"__{key[1].replace('/', '_')}"
    else:
        safe_name += f"__triplet{key}"
    out_path = output_dir / f"{safe_name}_reasoning.txt"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {out_path.name}")


def run_task(task):
    """Run the full visualization pipeline for one task."""
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

    # Score samples for wrong-proposal detection
    print("Scoring samples for wrong-proposal detection...")
    scored = score_all_samples(task, results, annots_list, annots_idx)

    # Report wrong-proposal statistics
    total_scored = len(scored)
    wp_cases = [(k, max(v.get('sft', (0, {}))[0], v.get('grpo', (0, {}))[0]))
                for k, v in scored.items()
                if max(v.get('sft', (0, {}))[0], v.get('grpo', (0, {}))[0]) >= WRONG_PROPOSAL_THRESHOLD]
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
    comp_dir   = OUTPUT_DIR / task / "comparison"
    sft_dir    = OUTPUT_DIR / task / "sft_detail"
    grpo_dir   = OUTPUT_DIR / task / "grpo_detail"
    reason_dir = OUTPUT_DIR / task / "reasoning"

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
            max_score = max(scored.get(key, {}).get(m, (0, {}))[0]
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
