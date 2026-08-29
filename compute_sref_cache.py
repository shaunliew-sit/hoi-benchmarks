#!/usr/bin/env python3
"""Phase 1 (run in conda env verl-tool-env): precompute SAHA-CF s_ref difficulty
buckets for every grounding sample and every referring pair, cache to JSON.

Grounding key = "file|||action|||object_category" -> bucket (MISS/PARTIAL/EASY).
Referring     = list of buckets in ann order (index == triplet_id).

Phase 2 (stratify_all_tasks.py, run in .venv) reads this cache and needs no verltool.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/AdaTooler-V/verltool/examples/data_preprocess/hoi")
from proposal_anchor import build_proposal_anchor_grounding  # noqa: E402
from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding  # noqa: E402

ANN_GROUND = {
    "hico": "/workspace/Groma/groma_data/benchmarks_simplified/hico_ground_test_simplified.json",
    "swig": "/workspace/Groma/groma_data/benchmarks_simplified/swig_ground_test_simplified.json",
}
ANN_REFER = {
    "hico": "/workspace/Groma/groma_data/benchmarks_simplified/hico_action_referring_test_simplified.json",
    "swig": "/workspace/Groma/groma_data/benchmarks_simplified/swig_action_referring_test_simplified.json",
}
PROPOSALS_DIR = "/workspace/hoi-tool-use-checkpoints/test_proposals"


def norm1000(b, w, h):
    return [b[0] * 1000.0 / w, b[1] * 1000.0 / h, b[2] * 1000.0 / w, b[3] * 1000.0 / h]


def bucket_of(s):
    return "MISS" if s < 1e-6 else ("PARTIAL" if s < 0.5 else "EASY")


def props_1000(stem):
    p = os.path.join(PROPOSALS_DIR, stem + ".json")
    if not os.path.exists(p):
        return []
    return [{"bbox_2d": q["bbox_1000"]} for q in json.load(open(p)).get("proposals", []) if q.get("bbox_1000")]


def sref(stem, flat, num_pairs):
    anchor = build_proposal_anchor_grounding(props_1000(stem), flat, num_pairs)
    return float(compute_sref_grounding(anchor, {"boxes_1000": flat, "num_pairs": num_pairs}))


def main():
    cache = {"grounding": {}, "referring": {}}
    for ds, path in ANN_GROUND.items():
        ann = json.load(open(path))
        m = {}
        for s in ann:
            w, h, boxes, inds, np_ = s["width"], s["height"], s["boxes"], s["gt_box_inds"], int(s["num_pairs"])
            flat = []
            for i in range(np_):
                flat.append(norm1000(boxes[inds[2 * i]], w, h))
                flat.append(norm1000(boxes[inds[2 * i + 1]], w, h))
            b = bucket_of(sref(os.path.splitext(s["file_name"])[0], flat, np_))
            m[f"{s['file_name']}|||{s['action']}|||{s['object_category']}"] = b
        cache["grounding"][ds] = m
        dist = {k: sum(1 for v in m.values() if v == k) for k in ("MISS", "PARTIAL", "EASY")}
        print(f"grounding {ds}: {len(m)} samples  {dist}")
    for ds, path in ANN_REFER.items():
        ann = json.load(open(path))
        # keyed by (file_stem, person_box, object_box) so any model's per_triplet file
        # joins by the GT pair regardless of row order / field names. Also keep an
        # index list for back-compat with index-aligned files.
        m, buckets = {}, []
        for s in ann:
            w, h, boxes = s["width"], s["height"], s["boxes"]
            pb, ob = boxes[s["person_box_idx"]], boxes[s["object_box_idx"]]
            flat = [norm1000(pb, w, h), norm1000(ob, w, h)]
            b = bucket_of(sref(os.path.splitext(s["file_name"])[0], flat, 1))
            stem = os.path.splitext(s["file_name"])[0]
            key = f"{stem}|||{','.join(str(int(v)) for v in pb)}|||{','.join(str(int(v)) for v in ob)}"
            m[key] = b
            buckets.append(b)
        cache["referring"][ds] = {"by_key": m, "by_index": buckets}
        dist = {k: buckets.count(k) for k in ("MISS", "PARTIAL", "EASY")}
        print(f"referring {ds}: {len(buckets)} pairs ({len(m)} unique keys)  {dist}")
    out = "/workspace/hoi-benchmarks/sref_cache.json"
    json.dump(cache, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
