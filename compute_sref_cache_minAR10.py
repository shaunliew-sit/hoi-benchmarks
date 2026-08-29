#!/usr/bin/env python3
"""Build a s_ref difficulty cache under the EVAL-ALIGNED minAR10 metric
(min(person,object) IoU, 10 thresholds 0.5..0.95) — parallel to the avg2
sref_cache.json — and report how HARD/EASY buckets shift vs avg2.

Self-contained (stdlib only). Reads paths.json (ann_ground + proposals_dir).
Writes sref_cache_minAR10.json (same structure as sref_cache.json).
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def iou(b1, b2):
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4: return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1]); x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2]-b1[0])*max(0.0, b1[3]-b1[1]); a2 = max(0.0, b2[2]-b2[0])*max(0.0, b2[3]-b2[1])
    u = a1 + a2 - inter; return inter / u if u > 0 else 0.0


def norm1000(b, w, h):
    return [b[0]*1000.0/w, b[1]*1000.0/h, b[2]*1000.0/w, b[3]*1000.0/h]


def props_1000(stem, pdir):
    p = os.path.join(pdir, stem + ".json")
    if not os.path.exists(p): return []
    return [q["bbox_1000"] for q in json.load(open(p)).get("proposals", []) if q.get("bbox_1000") and len(q["bbox_1000"]) == 4]


def anchor(props, flat, npairs):
    """Best proposal (by IoU) for each GT person/object box."""
    if not props or npairs <= 0: return []
    out = []
    for i in range(npairs):
        if 2*i+1 >= len(flat): break
        gp, go = flat[2*i], flat[2*i+1]
        out.append([max(props, key=lambda b: iou(b, gp)), max(props, key=lambda b: iou(b, go))])
    return out


def match_min(pred, gt, thr):
    if not pred or not gt: return 0
    sc = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            sc.append((min(iou(p[0], g[0]), iou(p[1], g[1])), pi, gi))
    sc.sort(reverse=True); mp, mg, c = set(), set(), 0
    for s, pi, gi in sc:
        if pi in mp or gi in mg: continue
        if s >= thr: c += 1; mp.add(pi); mg.add(gi)
    return c


def match_avg(pred, gt, thr):
    if not pred or not gt: return 0
    sc = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            sc.append((0.5*iou(p[0], g[0])+0.5*iou(p[1], g[1]), pi, gi))
    sc.sort(reverse=True); mp, mg, c = set(), set(), 0
    for s, pi, gi in sc:
        if pi in mp or gi in mg: continue
        if s >= thr: c += 1; mp.add(pi); mg.add(gi)
    return c


def sref_minAR10(anc, gt):
    if not anc or not gt: return 0.0
    thr = [0.5 + 0.05*i for i in range(10)]
    return sum(match_min(anc, gt, t)/len(gt) for t in thr)/10


def sref_avg2(anc, gt):
    if not anc or not gt: return 0.0
    return (match_avg(anc, gt, 0.5)/len(gt) + match_avg(anc, gt, 0.75)/len(gt))/2


def bucket(s):
    return "MISS" if s < 1e-6 else ("PARTIAL" if s < 0.5 else "EASY")


def main():
    cfg = json.load(open(os.path.join(HERE, "hoi_stratify_bundle", "paths.json")))
    ann_ground, pdir = cfg["ann_ground"], cfg["proposals_dir"]
    cache = {"grounding": {}, "referring": {}}
    flip = {}
    for ds, path in ann_ground.items():
        ann = json.load(open(path)); m = {}; counts = {"avg2": [0,0,0], "min": [0,0,0]}; e2h = 0; h2e = 0
        for s in ann:
            w, h, boxes, inds, np_ = s["width"], s["height"], s["boxes"], s["gt_box_inds"], int(s["num_pairs"])
            flat = []
            for i in range(np_):
                flat.append(norm1000(boxes[inds[2*i]], w, h)); flat.append(norm1000(boxes[inds[2*i+1]], w, h))
            gt = [[flat[2*i], flat[2*i+1]] for i in range(np_) if 2*i+1 < len(flat)]
            anc = anchor(props_1000(os.path.splitext(s["file_name"])[0], pdir), flat, np_)
            sa, sm = sref_avg2(anc, gt), sref_minAR10(anc, gt)
            ba, bm = bucket(sa), bucket(sm)
            m[f"{s['file_name']}|||{s['action']}|||{s['object_category']}"] = bm
            for i, b in enumerate(("MISS", "PARTIAL", "EASY")):
                if ba == b: counts["avg2"][i] += 1
                if bm == b: counts["min"][i] += 1
            ha = sa < 0.5; hm = sm < 0.5
            if (not ha) and hm: e2h += 1
            if ha and (not hm): h2e += 1
        cache["grounding"][ds] = m
        n = len(ann)
        ha = counts["avg2"][0]+counts["avg2"][1]; hm = counts["min"][0]+counts["min"][1]
        flip[ds] = (n, ha, hm, e2h, h2e)
        print(f"[{ds}] N={n}  HARD avg2={ha} ({ha/n:.1%})  HARD minAR10={hm} ({hm/n:.1%})  | EASY->HARD {e2h}, HARD->EASY {h2e}")
    json.dump(cache, open(os.path.join(HERE, "sref_cache_minAR10.json"), "w"))
    print("wrote sref_cache_minAR10.json")


if __name__ == "__main__":
    main()
