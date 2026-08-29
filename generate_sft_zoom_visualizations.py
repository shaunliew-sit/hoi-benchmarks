#!/usr/bin/env python3
"""
SFT vs GRPO Zoom Behavior Visualization Generator.

Shows GT | SFT | GRPO comparison focused on small-object zoom behavior.

Sources:
  SFT  → results-sft-qwen3vl-4b/
  GRPO → results-sft/

Run with: uv run python generate_sft_zoom_visualizations.py

Outputs: visualizations-sft/{task}/{comparison,sft_detail,grpo_detail,reasoning}/
"""
import json
import textwrap
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.resolve()
HICO_IMAGES = Path("/workspace/data/hico_20160224_det/images/test2015")
SWIG_IMAGES = Path("/workspace/data/swig_hoi/images_512")
ANNOT_DIR   = Path("/workspace/Groma/groma_data/benchmarks")
OUTPUT_DIR  = REPO / "visualizations-sft"

# ── Result files ──────────────────────────────────────────────────────────────
RESULTS_SFT = {
    "hico_ground": {
        "main":     REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909.json",
        "thinking": REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909_thinking.jsonl",
    },
    "hico_refer": {
        "per_triplet": REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_per_triplet.json",
        "thinking":    REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_thinking.jsonl",
    },
    "swig_ground": {
        "main":     REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937.json",
        "thinking": REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937_thinking.jsonl",
    },
    "swig_refer": {
        "per_triplet": REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_per_triplet.json",
        "thinking":    REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_thinking.jsonl",
    },
}

RESULTS_GRPO = {
    "hico_ground": {
        "main":     REPO / "results-sft/hico_ground_sft_new/hico_ground_sft_results_20260311_025017.json",
        "thinking": REPO / "results-sft/hico_ground_sft_new/hico_ground_sft_results_20260311_025017_thinking.jsonl",
    },
    "hico_refer": {
        "per_triplet": REPO / "results-sft/hico_action_sft_new/hico_action_sft_results_20260310_141500_per_triplet.json",
        "thinking":    REPO / "results-sft/hico_action_sft_new/hico_action_sft_results_20260310_141500_thinking.jsonl",
    },
    "swig_ground": {
        "main":     REPO / "results-sft/swig_ground_sft_new/swig_ground_sft_results_20260311_092108.json",
        "thinking": REPO / "results-sft/swig_ground_sft_new/swig_ground_sft_results_20260311_092108_thinking.jsonl",
    },
    "swig_refer": {
        "per_triplet": REPO / "results-sft/swig_action_sft_new/swig_action_sft_results_20260310_141513_per_triplet.json",
        "thinking":    REPO / "results-sft/swig_action_sft_new/swig_action_sft_results_20260310_141513_thinking.jsonl",
    },
}

ANNOT_FILES = {
    "hico_ground": ANNOT_DIR / "hico_ground_test.json",
    "hico_refer":  ANNOT_DIR / "hico_action_referring_test.json",
    "swig_ground": ANNOT_DIR / "swig_ground_test.json",
    "swig_refer":  ANNOT_DIR / "swig_action_referring_test.json",
}

# ── Mandatory images ───────────────────────────────────────────────────────────
MANDATORY_STEMS = [
    "HICO_test2015_00003584",
    "HICO_test2015_00006612",
    "HICO_test2015_00006692",
]

# ── Selection thresholds ──────────────────────────────────────────────────────
SMALL_OBJ_GROUND_PCT = 5.0
SMALL_OBJ_REFER_PCT  = 3.0
COMPLEX_PAIRS_THRESH = 3
IMAGES_PER_TASK      = 10

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "person":   "#FF4444",
    "object":   "#4444FF",
    "gt":       "#22AA22",
    "pred":     "#AA22FF",
    "zoom_box": "#FFD700",
}


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_output_dirs():
    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        for subdir in ["comparison", "sft_detail", "grpo_detail", "reasoning"]:
            (OUTPUT_DIR / task / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Output dirs created under {OUTPUT_DIR}")


def verify_paths():
    missing = []
    for label, d in [("HICO_IMAGES", HICO_IMAGES), ("SWIG_IMAGES", SWIG_IMAGES)]:
        if not d.exists():
            missing.append(f"  [dir][{label}]: {d}")
    for task, paths in {**{f"sft/{k}": v for k, v in RESULTS_SFT.items()},
                        **{f"grpo/{k}": v for k, v in RESULTS_GRPO.items()}}.items():
        for key, p in paths.items():
            if not p.exists():
                missing.append(f"  [{task}][{key}]: {p}")
    for task, p in ANNOT_FILES.items():
        if not p.exists():
            missing.append(f"  [annot][{task}]: {p}")
    if missing:
        raise FileNotFoundError("Fix missing files:\n" + "\n".join(missing))
    print("All source files verified OK")


# ── Coordinate utilities ──────────────────────────────────────────────────────

def scale_to_pixel(bbox_1000, width, height):
    x1, y1, x2, y2 = bbox_1000
    return [
        max(0, int(x1 * width / 1000)),
        max(0, int(y1 * height / 1000)),
        min(width - 1,  int(x2 * width / 1000)),
        min(height - 1, int(y2 * height / 1000)),
    ]


def bbox_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Answer parser ─────────────────────────────────────────────────────────────

def parse_ground_answer_sft(answer, width, height):
    """Parse SFT/GRPO grounding answer → list of (person_bbox_px, object_bbox_px)."""
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
                 o.get('label', '').lower() in ('person', 'human', 'man', 'woman',
                                                'player', 'rider', 'athlete')),
                None
            )
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


# ── Image loader ──────────────────────────────────────────────────────────────

def load_image(file_name, task):
    img_path = (HICO_IMAGES if 'hico' in task else SWIG_IMAGES) / file_name
    if not img_path.exists():
        for ext in ('.jpg', '.jpeg', '.png'):
            alt = img_path.with_suffix(ext)
            if alt.exists():
                return Image.open(alt).convert('RGB')
        raise FileNotFoundError(f"Image not found: {img_path}")
    return Image.open(img_path).convert('RGB')


# ── Drawing utilities ─────────────────────────────────────────────────────────

def draw_bboxes_on_ax(ax, img_array, bbox_groups, title, title_color='black', fontsize=9):
    """
    Draw image with colored bounding boxes.
    Pair format: bboxes = [[p_bbox, o_bbox], ...]
    Single format: bboxes = [[x1,y1,x2,y2], ...]
    """
    ax.imshow(img_array)
    ax.set_title(title, color=title_color, fontsize=fontsize,
                 fontweight='bold', pad=4, wrap=True)
    ax.axis('off')
    legend_patches = []
    for group in bbox_groups:
        bboxes  = group.get('bboxes', [])
        p_color = group.get('p_color', COLORS['pred'])
        o_color = group.get('o_color', COLORS['pred'])
        s_color = group.get('color', p_color)
        label   = group.get('label', '')
        ls      = group.get('linestyle', '-')
        lw      = group.get('linewidth', 2.0)
        if not bboxes:
            continue
        is_pair = isinstance(bboxes[0][0], (list, tuple))
        for item in bboxes:
            if is_pair:
                if len(item) < 2:
                    continue
                p_bb, o_bb = item[0], item[1]
                for bb, color in [(p_bb, p_color), (o_bb, o_color)]:
                    x1, y1, x2, y2 = bb
                    ax.add_patch(patches.Rectangle(
                        (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                        linewidth=lw, edgecolor=color, facecolor='none', linestyle=ls))
                ax.plot([(p_bb[0]+p_bb[2])/2, (o_bb[0]+o_bb[2])/2],
                        [(p_bb[1]+p_bb[3])/2, (o_bb[1]+o_bb[3])/2],
                        color=p_color, lw=1, ls='--', alpha=0.6)
            else:
                x1, y1, x2, y2 = item
                ax.add_patch(patches.Rectangle(
                    (x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                    linewidth=lw, edgecolor=s_color, facecolor='none', linestyle=ls))
        legend_color = s_color if not is_pair else p_color
        if label:
            legend_patches.append(patches.Patch(color=legend_color, label=label))
    if legend_patches:
        ax.legend(handles=legend_patches, loc='upper right', fontsize=6,
                  framealpha=0.7, markerscale=0.8)


def wrap_thinking(text, max_lines=4, line_width=60):
    if not text:
        return "(no thinking)"
    first_para = text.split('\n\n')[0].replace('\n', ' ').strip()
    wrapped = textwrap.fill(first_para, width=line_width)
    lines = wrapped.split('\n')
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:max(0, line_width - 3)] + '...'
    return '\n'.join(lines)


def get_zoom_crops(img, tool_calls):
    """Re-generate zoom crops from tool_calls list."""
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
        if x2 > x1 and y2 > y1:
            crops.append((tc.get('turn', len(crops)), bbox, img.crop((x1, y1, x2, y2))))
    return crops


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_ground_thinking_jsonl(path):
    """→ {(file_name, action, object): thinking_str}"""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (obj.get('file_name', ''), obj.get('action', ''), obj.get('object', ''))
                result[key] = obj.get('thinking', '')
            except json.JSONDecodeError:
                continue
    return result


def _load_refer_thinking_jsonl(path):
    """→ {(file_name, gt_action): thinking_str}  (first-wins on duplicate keys)"""
    result = {}
    if not path.exists():
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = (obj.get('file_name', ''), obj.get('gt_action', ''))
                if key not in result:
                    result[key] = obj.get('thinking', '')
            except json.JSONDecodeError:
                continue
    return result


def _index_ground_json(path):
    """→ {(file_name, 'action_object'): entry}"""
    with open(path) as f:
        data = json.load(f)
    idx = {}
    for entry in data:
        fn   = entry.get('file_name', '')
        aoid = f"{entry.get('action', '')}_{entry.get('object', '')}"
        idx[(fn, aoid)] = entry
    return idx


def _index_refer_json(path):
    """→ {int(triplet_id): entry}"""
    with open(path) as f:
        data = json.load(f)
    idx = {}
    for i, entry in enumerate(data):
        tid = entry.get('triplet_id', i)
        idx[int(tid)] = entry
    return idx


def load_all_results(task):
    """
    Load SFT and GRPO results for a task.
    Returns dict with keys: sft, grpo, sft_thinking, grpo_thinking
    """
    is_ground = 'ground' in task
    sft_paths  = RESULTS_SFT[task]
    grpo_paths = RESULTS_GRPO[task]

    if is_ground:
        print(f"  Loading SFT ground results ...")
        sft_idx  = _index_ground_json(sft_paths['main'])
        print(f"  Loading GRPO ground results ...")
        grpo_idx = _index_ground_json(grpo_paths['main'])
        sft_thinking  = _load_ground_thinking_jsonl(sft_paths['thinking'])
        grpo_thinking = _load_ground_thinking_jsonl(grpo_paths['thinking'])
    else:
        print(f"  Loading SFT refer results ...")
        sft_idx  = _index_refer_json(sft_paths['per_triplet'])
        print(f"  Loading GRPO refer results ...")
        grpo_idx = _index_refer_json(grpo_paths['per_triplet'])
        sft_thinking  = _load_refer_thinking_jsonl(sft_paths['thinking'])
        grpo_thinking = _load_refer_thinking_jsonl(grpo_paths['thinking'])

    return {
        "sft":          sft_idx,
        "grpo":         grpo_idx,
        "sft_thinking": sft_thinking,
        "grpo_thinking": grpo_thinking,
    }


def load_annotation(task):
    with open(ANNOT_FILES[task]) as f:
        annots = json.load(f)
    if 'ground' in task:
        idx = {}
        for entry in annots:
            fn   = entry['file_name']
            aoid = f"{entry.get('action', '')}_{entry.get('object_category', '')}"
            idx[(fn, aoid)] = entry
        return annots, idx
    else:
        return annots, {i: entry for i, entry in enumerate(annots)}


# ── GT extraction ─────────────────────────────────────────────────────────────

def extract_gt_pairs_pixel(annot_entry):
    boxes = annot_entry['boxes']
    box_inds = annot_entry['conversation'][1].get('box_inds') or []
    pairs = []
    for i in range(0, len(box_inds) - 1, 2):
        pi, oi = box_inds[i], box_inds[i + 1]
        if pi < len(boxes) and oi < len(boxes):
            pairs.append((boxes[pi], boxes[oi]))
    return pairs


def extract_refer_boxes_pixel(annot_entry):
    boxes    = annot_entry['boxes']
    box_inds = annot_entry['conversation'][0].get('box_inds') or [0, 1]
    p_bb = boxes[box_inds[0]] if box_inds[0] < len(boxes) else [0, 0, 1, 1]
    o_bb = boxes[box_inds[1]] if len(box_inds) > 1 and box_inds[1] < len(boxes) else [0, 0, 1, 1]
    return p_bb, o_bb


def get_gt_action_refer(annot_entry):
    return annot_entry['conversation'][1].get('value', '')


# ── Object area ───────────────────────────────────────────────────────────────

def _bbox_area_pct(bbox_px, W, H):
    x1, y1, x2, y2 = bbox_px
    return max(0, x2 - x1) * max(0, y2 - y1) / max(1, W * H) * 100.0


def _min_obj_pct_ground(annot_entry):
    W, H = annot_entry.get('width', 1), annot_entry.get('height', 1)
    pairs = extract_gt_pairs_pixel(annot_entry)
    return min((_bbox_area_pct(o, W, H) for _, o in pairs), default=100.0)


def _obj_pct_refer(annot_entry):
    W, H = annot_entry.get('width', 1), annot_entry.get('height', 1)
    _, o_bb = extract_refer_boxes_pixel(annot_entry)
    return _bbox_area_pct(o_bb, W, H)


# ── Thinking lookup ───────────────────────────────────────────────────────────

def _think_ground(results, model, fn, action, obj_cat):
    key = 'sft_thinking' if model == 'sft' else 'grpo_thinking'
    return results[key].get((fn, action, obj_cat), '')


def _think_refer(results, model, entry, fn, gt_action):
    thinking = entry.get('thinking_content', '')
    if not thinking:
        key = 'sft_thinking' if model == 'sft' else 'grpo_thinking'
        thinking = results[key].get((fn, gt_action), '')
    return thinking


# ── Sample scoring ────────────────────────────────────────────────────────────

def _score_ground(result_entry, annot_entry):
    score, reasons = 0, []
    n_zoom = sum(1 for tc in (result_entry.get('tool_calls') or [])
                 if tc.get('name') == 'zoom_in')
    if n_zoom:
        score += 4
        reasons.append(f"zoom_in_x{n_zoom}")
    if annot_entry is not None:
        pct = _min_obj_pct_ground(annot_entry)
        if pct < SMALL_OBJ_GROUND_PCT:
            score += 3
            reasons.append(f"small_obj_{pct:.1f}pct")
        n_pairs = annot_entry.get('num_pairs', len(extract_gt_pairs_pixel(annot_entry)))
        if n_pairs >= COMPLEX_PAIRS_THRESH:
            score += 2
            reasons.append(f"complex_{n_pairs}pairs")
    n_pred = result_entry.get('num_pred_pairs', 0)
    n_gt   = result_entry.get('num_gt_pairs', 0)
    if n_gt > 0 and n_pred != n_gt:
        score += 1
        reasons.append(f"mismatch_p{n_pred}_g{n_gt}")
    return score, reasons


def _score_refer(result_entry, annot_entry):
    score, reasons = 0, []
    n_zoom = sum(1 for tc in (result_entry.get('tool_calls') or [])
                 if tc.get('name') == 'zoom_in')
    if n_zoom:
        score += 4
        reasons.append(f"zoom_in_x{n_zoom}")
    if annot_entry is not None:
        pct = _obj_pct_refer(annot_entry)
        if pct < SMALL_OBJ_REFER_PCT:
            score += 3
            reasons.append(f"small_obj_{pct:.1f}pct")
    exact = result_entry.get('exact_match', True)
    if isinstance(exact, str):
        exact = exact.lower() == 'true'
    if not exact:
        score += 2
        reasons.append('wrong_pred')
    return score, reasons


# ── Sample selection ──────────────────────────────────────────────────────────

def select_samples(task, results, annots_idx, n=IMAGES_PER_TASK):
    """
    Select n samples prioritising:
      1. Mandatory HICO stems (HICO tasks only, highest-scoring entry per stem)
      2. High-scoring candidates (zoom/small-obj/complex), one per unique image stem
      3. Random fill (seed=42)
    Scoring is based on GRPO results (newer, more interesting zoom behaviour).
    Returns (selected_keys, all_scored_dict).
    """
    is_ground = 'ground' in task
    # Use GRPO index as primary for scoring (GRPO is the more recent model)
    idx = results['grpo']

    def get_stem(key):
        if is_ground:
            return Path(key[0]).stem
        e = idx.get(key, {})
        return Path(e.get('file_name', '')).stem

    def get_annot(key):
        if is_ground:
            return annots_idx.get(key)
        return annots_idx.get(int(key) if not isinstance(key, int) else key)

    all_scored = {}
    for key, entry in idx.items():
        annot = get_annot(key)
        if is_ground:
            s, r = _score_ground(entry, annot)
        else:
            s, r = _score_refer(entry, annot)
        all_scored[key] = (s, r)

    selected, selected_set, selected_stems = [], set(), set()

    # Phase 1: mandatory HICO stems
    if 'hico' in task:
        for stem in MANDATORY_STEMS:
            cands = [(k, all_scored[k][0]) for k in idx
                     if get_stem(k) == stem and k not in selected_set]
            if cands:
                best = max(cands, key=lambda x: x[1])[0]
                selected.append(best)
                selected_set.add(best)
                selected_stems.add(stem)

    # Phase 2: high-scoring unique stems
    for key, (sc, _) in sorted(all_scored.items(), key=lambda x: -x[1][0]):
        if len(selected) >= n:
            break
        stem = get_stem(key)
        if key not in selected_set and stem not in selected_stems and sc > 0:
            selected.append(key)
            selected_set.add(key)
            selected_stems.add(stem)

    # Phase 3: random fill
    if len(selected) < n:
        rng = random.Random(42)
        all_keys = list(idx.keys())
        rng.shuffle(all_keys)
        for key in all_keys:
            if len(selected) >= n:
                break
            stem = get_stem(key)
            if key not in selected_set and stem not in selected_stems:
                selected.append(key)
                selected_set.add(key)
                selected_stems.add(stem)

    return selected[:n], all_scored


def build_manifest(task, selected_keys, all_scored, results, annots_idx):
    is_ground = 'ground' in task
    grpo_idx  = results['grpo']
    manifest  = []
    for key in selected_keys:
        e  = grpo_idx.get(key, {})
        fn = e.get('file_name', '')
        action = e.get('action', e.get('ground_truth', ''))
        obj_cat = e.get('object', '')
        sc, reasons = all_scored.get(key, (0, []))
        manifest.append({
            "key":          list(key) if isinstance(key, tuple) else key,
            "file_name":    fn,
            "action":       action,
            "object":       obj_cat,
            "grpo_score":   sc,
            "reasons":      reasons,
            "is_mandatory": Path(fn).stem in MANDATORY_STEMS,
        })
    out = OUTPUT_DIR / task / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"  Saved manifest: {out.name}")


# ── Figure: Type A — 3-panel comparison (GT | SFT | GRPO) ────────────────────

def generate_comparison_ground(task, key, results, annots_idx, output_dir):
    fn, aoid = key
    annot = annots_idx.get(key)
    if annot is None:
        print(f"  [SKIP] No annotation for {key}")
        return
    try:
        img = load_image(fn, task)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    img_arr = np.array(img)
    W, H    = img.size
    action  = annot.get('action', '')
    obj_cat = annot.get('object_category', '')
    gt_pairs = extract_gt_pairs_pixel(annot)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{fn}  —  {action} | {obj_cat}", fontsize=9, y=1.01)

    # GT panel
    draw_bboxes_on_ax(axes[0], img_arr,
                      [{"bboxes": [[p, o] for p, o in gt_pairs],
                        "p_color": COLORS["gt"], "o_color": COLORS["gt"],
                        "label": "GT pair"}],
                      f"Ground Truth\n{len(gt_pairs)} pair(s)", fontsize=8)

    for ax, model_label, model_key in [
        (axes[1], "SFT",  "sft"),
        (axes[2], "GRPO", "grpo"),
    ]:
        entry   = results[model_key].get(key, {})
        think   = _think_ground(results, model_key, fn, action, obj_cat)
        preds   = parse_ground_answer_sft(entry.get('answer'), W, H)
        matched = entry.get('matches_per_threshold', {}).get('0.5', {}).get('matched', '?')
        n_tools = len(entry.get('tool_calls') or [])
        subtitle = (f"{model_label}\npred={len(preds)}, gt={len(gt_pairs)}, "
                    f"matched@0.5={matched}  [tools: {n_tools}]\n"
                    f"{wrap_thinking(think, max_lines=2, line_width=50)}")
        draw_bboxes_on_ax(ax, img_arr,
                          [{"bboxes": [[p, o] for p, o in preds],
                            "p_color": COLORS["person"], "o_color": COLORS["object"],
                            "label": "Pred pair"}],
                          subtitle, fontsize=7)

    plt.tight_layout()
    safe = Path(fn).stem + f"__{aoid.replace('/', '_').replace(' ', '_')}"
    out  = output_dir / f"{safe}_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


def generate_comparison_refer(task, key, results, annots_idx, output_dir):
    annot = annots_idx.get(int(key) if not isinstance(key, int) else key)
    if annot is None:
        print(f"  [SKIP] No annotation for triplet_id={key}")
        return
    fn = annot['file_name']
    try:
        img = load_image(fn, task)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    img_arr   = np.array(img)
    gt_action = get_gt_action_refer(annot)
    p_bb, o_bb = extract_refer_boxes_pixel(annot)
    input_groups = [
        {"bboxes": [p_bb], "color": COLORS["person"], "label": "Person", "linewidth": 2.5},
        {"bboxes": [o_bb], "color": COLORS["object"],  "label": "Object",  "linewidth": 2.5},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{fn}  —  GT: {gt_action}", fontsize=9, y=1.01)

    draw_bboxes_on_ax(axes[0], img_arr, input_groups,
                      f"Ground Truth\n{gt_action}", fontsize=8)

    for ax, model_label, model_key in [
        (axes[1], "SFT",  "sft"),
        (axes[2], "GRPO", "grpo"),
    ]:
        entry = results[model_key].get(key, {})
        think = _think_refer(results, model_key, entry, fn, gt_action)
        pred  = entry.get('prediction', '—')
        exact = entry.get('exact_match', False)
        if isinstance(exact, str):
            exact = exact.lower() == 'true'
        n_tools = len(entry.get('tool_calls') or [])
        t_color  = '#22AA22' if exact else '#CC2222'
        subtitle = (f"{model_label}\nPred: {pred}  [tools: {n_tools}]\n"
                    f"{wrap_thinking(think, max_lines=2, line_width=50)}")
        draw_bboxes_on_ax(ax, img_arr, input_groups, subtitle,
                          title_color=t_color, fontsize=7)

    plt.tight_layout()
    stem = Path(fn).stem
    out  = output_dir / f"{stem}__t{key}_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


# ── Figure: Type B — detail with zoom crops ───────────────────────────────────

def _detail_ground(task, key, model_label, model_key, results, annots_idx, output_dir):
    fn, aoid = key
    annot = annots_idx.get(key)
    if annot is None:
        return
    try:
        img = load_image(fn, task)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    img_arr  = np.array(img)
    W, H     = img.size
    action   = annot.get('action', '')
    obj_cat  = annot.get('object_category', '')
    gt_pairs = extract_gt_pairs_pixel(annot)
    entry    = results[model_key].get(key, {})
    think    = _think_ground(results, model_key, fn, action, obj_cat)
    preds    = parse_ground_answer_sft(entry.get('answer'), W, H)
    tool_calls = entry.get('tool_calls') or []
    zoom_crops = get_zoom_crops(img, tool_calls)
    n_crops    = len(zoom_crops)

    n_cols = max(2, n_crops)
    if n_crops > 0:
        fig = plt.figure(figsize=(6 * n_cols, 12))
        gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.5, wspace=0.3)
        ax_gt   = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
        crop_axes = [fig.add_subplot(gs[1, i]) for i in range(n_crops)]
    else:
        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(14, 6))
        crop_axes = []

    # GT panel + zoom highlights
    gt_groups = [{"bboxes": [[p, o] for p, o in gt_pairs],
                  "p_color": COLORS["gt"], "o_color": COLORS["gt"],
                  "label": "GT pair"}]
    for tc in tool_calls:
        if tc.get('name') == 'zoom_in':
            zb = tc.get('bbox', [])
            if len(zb) == 4:
                gt_groups.append({
                    "bboxes": [scale_to_pixel(zb, W, H)],
                    "color": COLORS["zoom_box"],
                    "label": f"zoom_in(t{tc['turn']})",
                    "linestyle": "--", "linewidth": 2.5,
                })
    obj_pct = _min_obj_pct_ground(annot)
    draw_bboxes_on_ax(ax_gt, img_arr, gt_groups,
                      f"GT + Zoom Targets\n{len(gt_pairs)} pair(s), obj {obj_pct:.1f}% img",
                      fontsize=8)

    # Prediction panel
    matched = entry.get('matches_per_threshold', {}).get('0.5', {}).get('matched', '?')
    m75     = entry.get('matches_per_threshold', {}).get('0.75', {}).get('matched', '?')
    pred_title = (f"{model_label}  pred={len(preds)}, gt={len(gt_pairs)}\n"
                  f"matched@0.5={matched}, @0.75={m75}\n"
                  f"{wrap_thinking(think, max_lines=2, line_width=52)}")
    draw_bboxes_on_ax(ax_pred, img_arr,
                      [{"bboxes": [[p, o] for p, o in preds],
                        "p_color": COLORS["person"], "o_color": COLORS["object"],
                        "label": "Pred pair"}],
                      pred_title, fontsize=7)

    for ax, (turn, bbox_1000, crop_img) in zip(crop_axes, zoom_crops):
        ax.imshow(np.array(crop_img))
        cw = (bbox_1000[2] - bbox_1000[0]) / 10.0
        ch = (bbox_1000[3] - bbox_1000[1]) / 10.0
        ax.set_title(f"Zoom turn={turn}  {cw:.0f}%W×{ch:.0f}%H", fontsize=7)
        ax.axis('off')

    tool_seq = ' → '.join(
        f"zoom_in({tc['bbox']})" if tc.get('name') == 'zoom_in' else tc.get('name', '?')
        for tc in tool_calls)
    fig.suptitle(f"{fn}  |  {action} | {obj_cat}  |  {model_label}", fontsize=9)
    if tool_seq:
        fig.text(0.5, 0.01, f"Tools: {tool_seq[:200]}", ha='center', fontsize=6, color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    safe = Path(fn).stem + f"__{aoid.replace('/', '_').replace(' ', '_')}"
    out  = output_dir / f"{safe}_{model_key}_detail.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


def _detail_refer(task, key, model_label, model_key, results, annots_idx, output_dir):
    annot = annots_idx.get(int(key) if not isinstance(key, int) else key)
    if annot is None:
        return
    fn = annot['file_name']
    try:
        img = load_image(fn, task)
    except FileNotFoundError as e:
        print(f"  [SKIP] {e}")
        return

    img_arr   = np.array(img)
    W, H      = img.size
    gt_action = get_gt_action_refer(annot)
    p_bb, o_bb = extract_refer_boxes_pixel(annot)
    entry      = results[model_key].get(key, {})
    pred       = entry.get('prediction', '—')
    exact      = entry.get('exact_match', False)
    if isinstance(exact, str):
        exact = exact.lower() == 'true'
    think      = _think_refer(results, model_key, entry, fn, gt_action)
    tool_calls = entry.get('tool_calls') or []
    zoom_crops = get_zoom_crops(img, tool_calls)
    n_crops    = len(zoom_crops)
    obj_pct    = _obj_pct_refer(annot)
    match_color = '#22AA22' if exact else '#CC2222'
    match_label = 'CORRECT' if exact else 'WRONG'

    # Object crop with 20% padding
    ox1, oy1, ox2, oy2 = o_bb
    px = max(5, int((ox2 - ox1) * 0.2))
    py = max(5, int((oy2 - oy1) * 0.2))
    obj_crop = img.crop((max(0, ox1-px), max(0, oy1-py),
                         min(W, ox2+px), min(H, oy2+py)))

    # Layout: left=full image, center=object crop, right=text, then crop panels below
    if n_crops > 0:
        n_cols = max(3, n_crops + 1)
        fig = plt.figure(figsize=(6 * n_cols, 12))
        gs  = gridspec.GridSpec(2, n_cols, figure=fig, hspace=0.5, wspace=0.3)
        ax_full  = fig.add_subplot(gs[0, 0])
        ax_obj   = fig.add_subplot(gs[0, 1])
        ax_text  = fig.add_subplot(gs[0, 2])
        crop_axes = [fig.add_subplot(gs[1, i]) for i in range(n_crops)]
    else:
        fig, (ax_full, ax_obj, ax_text) = plt.subplots(1, 3, figsize=(18, 7))
        crop_axes = []

    fig.suptitle(f"{fn}  |  triplet_id={key}  |  {model_label}", fontsize=9)

    # Full image + boxes
    bbox_groups = [
        {"bboxes": [p_bb], "color": COLORS["person"], "label": "Person", "linewidth": 2.5},
        {"bboxes": [o_bb], "color": COLORS["object"],  "label": "Object",  "linewidth": 2.5},
    ]
    for tc in tool_calls:
        if tc.get('name') == 'zoom_in':
            zb = tc.get('bbox', [])
            if len(zb) == 4:
                bbox_groups.append({
                    "bboxes": [scale_to_pixel(zb, W, H)],
                    "color": COLORS["zoom_box"], "label": "zoom_in",
                    "linestyle": "--", "linewidth": 2.0,
                })
    draw_bboxes_on_ax(ax_full, img_arr, bbox_groups,
                      f"Person + Object Input\nGT: {gt_action}", fontsize=8)

    # Object crop
    ax_obj.imshow(np.array(obj_crop))
    ax_obj.set_title(f"Object Region Zoom\n({obj_pct:.2f}% of image)", fontsize=8)
    ax_obj.axis('off')

    # Text panel
    ax_text.axis('off')
    tool_seq = ' → '.join(
        'zoom_in' if tc.get('name') == 'zoom_in' else tc.get('name', '?')
        for tc in tool_calls) or '(no tool calls)'
    text_content = (
        f"PREDICTION: {pred}\n"
        f"GT ACTION:  {gt_action}\n"
        f"MATCH:      {match_label}\n\n"
        f"TOOL CALLS:\n  {tool_seq}\n\n"
        f"THINKING:\n{wrap_thinking(think, max_lines=7, line_width=42)}"
    )
    ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes,
                 verticalalignment='top', fontsize=7.5, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax_text.set_title(f'{model_label} Analysis', fontsize=9,
                      color=match_color, fontweight='bold')

    for ax, (turn, bbox_1000, crop_img) in zip(crop_axes, zoom_crops):
        ax.imshow(np.array(crop_img))
        cw = (bbox_1000[2] - bbox_1000[0]) / 10.0
        ch = (bbox_1000[3] - bbox_1000[1]) / 10.0
        ax.set_title(f"Zoom turn={turn}  {cw:.0f}%W×{ch:.0f}%H", fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    stem = Path(fn).stem
    out  = output_dir / f"{stem}__t{key}_{model_key}_detail.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")


# ── Figure: Type C — reasoning text ──────────────────────────────────────────

def _save_reasoning_ground(task, key, results, annots_idx, output_dir):
    fn, aoid = key
    annot   = annots_idx.get(key)
    action  = annot.get('action', '') if annot else ''
    obj_cat = annot.get('object_category', '') if annot else ''

    lines = [f"=== Grounding Reasoning  [{task}] ===",
             f"File:   {fn}", f"Action: {action}", f"Object: {obj_cat}", ""]

    for model_label, model_key in [("SFT", "sft"), ("GRPO", "grpo")]:
        entry      = results[model_key].get(key, {})
        tool_calls = entry.get('tool_calls') or []
        matched    = entry.get('matches_per_threshold', {}).get('0.5', {}).get('matched', '?')
        think      = _think_ground(results, model_key, fn, action, obj_cat)
        lines += [f"--- {model_label} ---",
                  f"GT pairs:   {entry.get('num_gt_pairs', '?')}",
                  f"Pred pairs: {entry.get('num_pred_pairs', '?')}",
                  f"Matched@0.5: {matched}", "Tool calls:"]
        for tc in tool_calls:
            lines.append(f"  Turn {tc.get('turn','?')}: {tc.get('name','?')}"
                         + (f"({tc['bbox']})" if tc.get('name') == 'zoom_in' else ''))
        lines += ["Thinking:", think or "(no thinking)", ""]

    safe = Path(fn).stem + f"__{aoid.replace('/', '_').replace(' ', '_')}"
    (output_dir / f"{safe}_reasoning.txt").write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved: {safe}_reasoning.txt")


def _save_reasoning_refer(task, key, results, annots_idx, output_dir):
    annot     = annots_idx.get(int(key) if not isinstance(key, int) else key)
    fn        = annot['file_name'] if annot else f"triplet_{key}"
    gt_action = get_gt_action_refer(annot) if annot else ''

    lines = [f"=== Action Referring Reasoning  [{task}] ===",
             f"File:       {fn}", f"Triplet ID: {key}", f"GT action:  {gt_action}", ""]

    for model_label, model_key in [("SFT", "sft"), ("GRPO", "grpo")]:
        entry      = results[model_key].get(key, {})
        pred       = entry.get('prediction', '—')
        exact      = entry.get('exact_match', False)
        if isinstance(exact, str):
            exact = exact.lower() == 'true'
        tool_calls = entry.get('tool_calls') or []
        think      = _think_refer(results, model_key, entry, fn, gt_action)
        lines += [f"--- {model_label} ---",
                  f"Prediction: {pred}", f"Exact match: {exact}", "Tool calls:"]
        for tc in tool_calls:
            lines.append(f"  Turn {tc.get('turn','?')}: {tc.get('name','?')}"
                         + (f"({tc['bbox']})" if tc.get('name') == 'zoom_in' else ''))
        lines += ["Thinking:", think or "(no thinking)", ""]

    stem = Path(fn).stem
    (output_dir / f"{stem}__t{key}_reasoning.txt").write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved: {stem}__t{key}_reasoning.txt")


# ── Per-task orchestration ────────────────────────────────────────────────────

def run_task(task):
    print(f"\n{'=' * 60}\nTASK: {task}\n{'=' * 60}")
    is_ground = 'ground' in task

    print("  Loading annotations ...")
    annots_list, annots_idx = load_annotation(task)
    print(f"  {len(annots_list)} annotation entries")

    results = load_all_results(task)
    print(f"  SFT entries:  {len(results['sft'])}")
    print(f"  GRPO entries: {len(results['grpo'])}")

    print("  Selecting samples ...")
    selected_keys, all_scored = select_samples(task, results, annots_idx)
    print(f"  Selected {len(selected_keys)} samples:")
    for k in selected_keys:
        sc, r = all_scored.get(k, (0, []))
        label = f"{k[0]} / {k[1]}" if is_ground else f"triplet_id={k}"
        print(f"    score={sc:2d}  {r}  {label}")

    build_manifest(task, selected_keys, all_scored, results, annots_idx)

    cmp_dir  = OUTPUT_DIR / task / "comparison"
    sft_dir  = OUTPUT_DIR / task / "sft_detail"
    grpo_dir = OUTPUT_DIR / task / "grpo_detail"
    rsn_dir  = OUTPUT_DIR / task / "reasoning"

    print("\n  Generating comparison figures (GT | SFT | GRPO) ...")
    for key in selected_keys:
        try:
            if is_ground:
                generate_comparison_ground(task, key, results, annots_idx, cmp_dir)
            else:
                generate_comparison_refer(task, key, results, annots_idx, cmp_dir)
        except Exception as e:
            print(f"  [ERROR] comparison {key}: {e}")

    print("\n  Generating SFT detail figures ...")
    for key in selected_keys:
        try:
            if is_ground:
                _detail_ground(task, key, "SFT", "sft", results, annots_idx, sft_dir)
            else:
                _detail_refer(task, key, "SFT", "sft", results, annots_idx, sft_dir)
        except Exception as e:
            print(f"  [ERROR] sft_detail {key}: {e}")

    print("\n  Generating GRPO detail figures ...")
    for key in selected_keys:
        try:
            if is_ground:
                _detail_ground(task, key, "GRPO", "grpo", results, annots_idx, grpo_dir)
            else:
                _detail_refer(task, key, "GRPO", "grpo", results, annots_idx, grpo_dir)
        except Exception as e:
            print(f"  [ERROR] grpo_detail {key}: {e}")

    print("\n  Saving reasoning text files ...")
    for key in selected_keys:
        try:
            if is_ground:
                _save_reasoning_ground(task, key, results, annots_idx, rsn_dir)
            else:
                _save_reasoning_refer(task, key, results, annots_idx, rsn_dir)
        except Exception as e:
            print(f"  [ERROR] reasoning {key}: {e}")

    print(f"\n  Done: {task}")
    return selected_keys


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_output_dirs()
    verify_paths()
    print("\nSFT vs GRPO zoom visualization pipeline\n")

    all_selected = {}
    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        all_selected[task] = run_task(task)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for task, sel in all_selected.items():
        print(f"  {task}: {len(sel)} samples  "
              f"→ {len(sel)*3} figs (comparison + sft_detail + grpo_detail) + {len(sel)} txt")
    print(f"\n  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
