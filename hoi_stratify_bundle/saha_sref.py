"""Vendored SAHA-CF grounding s_ref functions (stdlib-only, no verltool/torch).

Copied verbatim from the training repo so the stratifier bundle is self-contained:
  - verl_tool/workers/reward_manager/sds_grpo.py  (calculate_iou, match_pairs_greedy,
        parse_grounding_answer, compute_grounding_outcome)
  - verl_tool/workers/reward_manager/saha_cf.py   (compute_sref_grounding)
  - examples/data_preprocess/hoi/proposal_anchor.py (iou, build_proposal_anchor_grounding)

s_ref = score of the proposal-only anchor answer vs the target GT
      = (recall@IoU0.5 + recall@IoU0.75) / 2, pair-matched greedily with
        pair_score = 0.5*person_iou + 0.5*object_iou.  GT-anchored, policy-independent.
"""
import json
import re


def iou(b1, b2):
    """IoU between two [x1,y1,x2,y2] boxes; 0.0 on malformed input (proposal_anchor.py)."""
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def build_proposal_anchor_grounding(proposals, boxes_1000, num_pairs):
    """Proposal-only answer: best proposal (by IoU) for each GT person/object."""
    if not proposals or num_pairs <= 0:
        return []
    boxes = [p["bbox_2d"] for p in proposals if p.get("bbox_2d") and len(p["bbox_2d"]) == 4]
    if not boxes:
        return []
    anchor = []
    for i in range(num_pairs):
        pidx, oidx = 2 * i, 2 * i + 1
        if oidx >= len(boxes_1000):
            break
        gt_p, gt_o = boxes_1000[pidx], boxes_1000[oidx]
        best_p = max(boxes, key=lambda b: iou(b, gt_p))
        best_o = max(boxes, key=lambda b: iou(b, gt_o))
        anchor.append([{"bbox_2d": best_p, "label": "person"},
                       {"bbox_2d": best_o, "label": "object"}])
    return anchor


def calculate_iou(box1, box2):
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def match_pairs_greedy(pred_pairs, gt_pairs, threshold):
    """Greedy pair match; pair_score = 0.5*person_iou + 0.5*object_iou (s_ref convention)."""
    if not pred_pairs or not gt_pairs:
        return 0
    scores = []
    for pi, pred in enumerate(pred_pairs):
        for gi, gt in enumerate(gt_pairs):
            person_iou = calculate_iou(pred[0]["bbox_2d"], gt[0]["bbox_2d"])
            object_iou = calculate_iou(pred[1]["bbox_2d"], gt[1]["bbox_2d"])
            scores.append((0.5 * person_iou + 0.5 * object_iou, pi, gi))
    scores.sort(key=lambda x: x[0], reverse=True)
    mp, mg, count = set(), set(), 0
    for score, pi, gi in scores:
        if pi in mp or gi in mg:
            continue
        if score >= threshold:
            count += 1
            mp.add(pi); mg.add(gi)
    return count


def parse_grounding_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    content = match.group(1).strip() if match else text.strip()
    pairs = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, list) and len(parsed) == 2 and all(
                    isinstance(d, dict) and isinstance(d.get("bbox_2d"), list) and len(d["bbox_2d"]) == 4
                    for d in parsed):
                pairs.append(parsed)
        except (json.JSONDecodeError, TypeError):
            continue
    return pairs


def compute_grounding_outcome(pred_text, gt_data):
    pred_pairs = parse_grounding_answer(pred_text)
    boxes_1000 = gt_data.get("boxes_1000", [])
    num_pairs = gt_data.get("num_pairs", 0)
    gt_pairs = []
    for i in range(num_pairs):
        if i * 2 + 1 < len(boxes_1000):
            gt_pairs.append([{"bbox_2d": boxes_1000[i * 2]}, {"bbox_2d": boxes_1000[i * 2 + 1]}])
    if not gt_pairs:
        return 1.0 if not pred_pairs else 0.0
    r05 = match_pairs_greedy(pred_pairs, gt_pairs, 0.5) / len(gt_pairs)
    r075 = match_pairs_greedy(pred_pairs, gt_pairs, 0.75) / len(gt_pairs)
    return (r05 + r075) / 2.0


def compute_sref_grounding(anchor, gt_data):
    """s_ref: score the proposal anchor as if it were the model's answer."""
    if anchor is None or len(anchor) == 0:
        return 0.0
    lines = "\n".join(
        json.dumps(pair, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        for pair in anchor)
    return compute_grounding_outcome(f"<answer>{lines}</answer>", gt_data)
