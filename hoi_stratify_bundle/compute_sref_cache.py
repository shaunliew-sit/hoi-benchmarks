#!/usr/bin/env python3
"""Phase 1 — precompute SAHA-CF s_ref difficulty buckets, cache to JSON.

Self-contained: vendored s_ref (saha_sref.py), no verltool needed. Reads dataset
paths from paths.json (next to this script, or --paths). Run once per test set;
the cache is model-independent.

Grounding cache: { "file|||action|||object": bucket } per dataset.
Referring cache: { "by_key": {"stem|||pbox|||obox": bucket}, "by_index": [bucket,...] }.
Buckets: MISS s_ref<1e-6 | PARTIAL 0<s_ref<0.5 | EASY s_ref>=0.5 ; HARD = s_ref<0.5.
"""
import argparse
import json
import os

from saha_sref import build_proposal_anchor_grounding, compute_sref_grounding

HERE = os.path.dirname(os.path.abspath(__file__))


def load_paths(p):
    cfg = json.load(open(p))
    return cfg["ann_ground"], cfg["ann_refer"], cfg["proposals_dir"], cfg.get("cache", "sref_cache.json")


def norm1000(b, w, h):
    return [b[0] * 1000.0 / w, b[1] * 1000.0 / h, b[2] * 1000.0 / w, b[3] * 1000.0 / h]


def bucket_of(s):
    return "MISS" if s < 1e-6 else ("PARTIAL" if s < 0.5 else "EASY")


def props_1000(stem, proposals_dir):
    p = os.path.join(proposals_dir, stem + ".json")
    if not os.path.exists(p):
        return []
    return [{"bbox_2d": q["bbox_1000"]} for q in json.load(open(p)).get("proposals", []) if q.get("bbox_1000")]


def sref(stem, flat, num_pairs, proposals_dir):
    anchor = build_proposal_anchor_grounding(props_1000(stem, proposals_dir), flat, num_pairs)
    return float(compute_sref_grounding(anchor, {"boxes_1000": flat, "num_pairs": num_pairs}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default=os.path.join(HERE, "paths.json"))
    args = ap.parse_args()
    ann_ground, ann_refer, proposals_dir, cache_path = load_paths(args.paths)
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(HERE, cache_path)
    cache = {"grounding": {}, "referring": {}}

    for ds, path in ann_ground.items():
        ann = json.load(open(path))
        m = {}
        for s in ann:
            w, h, boxes, inds, np_ = s["width"], s["height"], s["boxes"], s["gt_box_inds"], int(s["num_pairs"])
            flat = []
            for i in range(np_):
                flat.append(norm1000(boxes[inds[2 * i]], w, h))
                flat.append(norm1000(boxes[inds[2 * i + 1]], w, h))
            b = bucket_of(sref(os.path.splitext(s["file_name"])[0], flat, np_, proposals_dir))
            m[f"{s['file_name']}|||{s['action']}|||{s['object_category']}"] = b
        cache["grounding"][ds] = m
        dist = {k: sum(1 for v in m.values() if v == k) for k in ("MISS", "PARTIAL", "EASY")}
        print(f"grounding {ds}: {len(m)} samples  {dist}")

    for ds, path in ann_refer.items():
        ann = json.load(open(path))
        m, buckets = {}, []
        for s in ann:
            w, h, boxes = s["width"], s["height"], s["boxes"]
            pb, ob = boxes[s["person_box_idx"]], boxes[s["object_box_idx"]]
            flat = [norm1000(pb, w, h), norm1000(ob, w, h)]
            b = bucket_of(sref(os.path.splitext(s["file_name"])[0], flat, 1, proposals_dir))
            stem = os.path.splitext(s["file_name"])[0]
            key = f"{stem}|||{','.join(str(int(v)) for v in pb)}|||{','.join(str(int(v)) for v in ob)}"
            m[key] = b
            buckets.append(b)
        cache["referring"][ds] = {"by_key": m, "by_index": buckets}
        dist = {k: buckets.count(k) for k in ("MISS", "PARTIAL", "EASY")}
        print(f"referring {ds}: {len(buckets)} pairs ({len(m)} unique keys)  {dist}")

    json.dump(cache, open(cache_path, "w"))
    print("wrote", cache_path)


if __name__ == "__main__":
    main()
