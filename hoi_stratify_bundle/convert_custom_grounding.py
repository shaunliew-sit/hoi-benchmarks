#!/usr/bin/env python3
"""Convert Claude/GPT custom grounding result files into the standard per-sample
format the stratifier reads. Their files carry predicted_pairs + gt_pairs (pixel
boxes) but no matches_per_threshold, so we compute it with the SAME greedy pair
matching the eval uses (pair IoU = min(person_iou, object_iou), one-to-one).

Usage: python convert_custom_grounding.py IN.json OUT.json
Writes a list of {file_name, action, object, num_gt_pairs, num_pred_pairs,
matches_per_threshold} and prints the overall AR for validation.
"""
import json
import os
import sys

THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_greedy(pred, gt, thr):
    scores = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            v = min(iou(p[0], g[0]), iou(p[1], g[1]))
            if v >= thr:
                scores.append((v, pi, gi))
    scores.sort(reverse=True)
    up, ug = set(), set()
    for v, pi, gi in scores:
        if pi not in up and gi not in ug:
            up.add(pi); ug.add(gi)
    return len(ug)


def pairs(lst):
    out = []
    for d in lst or []:
        pb, ob = d.get("person_bbox"), d.get("object_bbox")
        if pb and ob and len(pb) == 4 and len(ob) == 4:
            out.append(([float(x) for x in pb], [float(x) for x in ob]))
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src))
    out = []
    tot = {t: {"tp": 0, "fn": 0} for t in THRESHOLDS}
    for it in data:
        pred = pairs(it.get("predicted_pairs"))
        gt = pairs(it.get("gt_pairs"))
        mpt = {}
        for t in THRESHOLDS:
            m = match_greedy(pred, gt, t)
            mpt[str(t)] = {"matched": m, "unmatched_preds": len(pred) - m, "unmatched_gts": len(gt) - m}
            tot[t]["tp"] += m; tot[t]["fn"] += len(gt) - m
        out.append({
            "file_name": os.path.basename(it.get("image_path", str(it.get("image_id")))),
            "action": it.get("action"),
            "object": it.get("object_category"),
            "num_gt_pairs": len(gt),
            "num_pred_pairs": len(pred),
            "matches_per_threshold": mpt,
        })
    json.dump(out, open(dst, "w"))
    recalls = [tot[t]["tp"] / (tot[t]["tp"] + tot[t]["fn"]) if (tot[t]["tp"] + tot[t]["fn"]) > 0 else 0 for t in THRESHOLDS]
    ar = 100 * sum(recalls) / len(THRESHOLDS)
    print(f"wrote {dst}  n={len(out)}  AR={ar:.2f}  AR@0.5={100*recalls[0]:.2f}  AR@0.75={100*recalls[5]:.2f}")


if __name__ == "__main__":
    main()
