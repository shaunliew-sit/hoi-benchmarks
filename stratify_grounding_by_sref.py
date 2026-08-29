#!/usr/bin/env python3
"""Stratify EXISTING grounding eval results into HARD / EASY by SAHA-CF s_ref.

No re-running of evaluation. We:
  1. Recompute the SAME grounding s_ref the reward used, per TEST sample:
       anchor = build_proposal_anchor_grounding(test_proposals, gt_boxes_1000, num_pairs)
       s_ref  = compute_sref_grounding(anchor, {boxes_1000, num_pairs})
     -> sample-intrinsic difficulty (how good the detector proposal is vs GT),
        independent of which model we then score.
  2. Bucket: MISS s_ref==0 | PARTIAL 0<s_ref<0.5 | EASY s_ref>=0.5 ; HARD = MISS+PARTIAL (s_ref<0.5).
  3. For each model's per-sample grounding result JSON (which stores
     matches_per_threshold = {thr: {matched, unmatched_gts, ...}}), recompute
     AR / AR@0.5 / AR@0.75 within each bucket using the eval's exact MICRO formula:
       recall_t = sum(matched) / sum(matched + unmatched_gts)   over samples in bucket
       AR       = mean_t recall_t   for t in 0.50..0.95 step 0.05

Run from /workspace/hoi-benchmarks in conda env verl-tool-env.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/workspace/AdaTooler-V/verltool/examples/data_preprocess/hoi")
from proposal_anchor import build_proposal_anchor_grounding  # noqa: E402
from verl_tool.workers.reward_manager.saha_cf import compute_sref_grounding  # noqa: E402

ANN = {
    "hico": "/workspace/Groma/groma_data/benchmarks_simplified/hico_ground_test_simplified.json",
    "swig": "/workspace/Groma/groma_data/benchmarks_simplified/swig_ground_test_simplified.json",
}
PROPOSALS_DIR = "/workspace/hoi-tool-use-checkpoints/test_proposals"
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]

# model result JSONs (row -> dataset -> path). +proposal row was never evaluated.
RESULTS = {
    "base_8B": {
        "hico": "/workspace/Groma/results-redo/hico_ground_qwen3vl_8B/hico_ground_qwen3vl_results_20251104_025034.json",
        "swig": "/workspace/Groma/results-redo/swig_ground_qwen3vl-8B/swig_ground_qwen3vl_results_20251104_025324.json",
    },
    "sft": {
        "hico": "/workspace/hoi-benchmarks/results-sftgrpo/hico_ground_sft/hico_ground_sft_results_20260622_095927.json",
        "swig": "/workspace/hoi-benchmarks/results-sftgrpo/swig_ground_sft/swig_ground_sft_results_20260622_095920.json",
    },
    "grpo": {
        "hico": "/workspace/hoi-benchmarks/results-sftgrpo/hico_ground_grpo/hico_ground_sft_results_20260621_102754.json",
        "swig": "/workspace/hoi-benchmarks/results-sftgrpo/swig_ground_grpo/swig_ground_sft_results_20260621_102826.json",
    },
}
ROW_ORDER = ["base_8B", "sft", "grpo"]
ROW_LABEL = {"base_8B": "Qwen3-VL-8B (base)", "sft": "+proposal+SFT", "grpo": "Full (+SFT+GRPO)"}


def norm_box(b, w, h):
    return [b[0] * 1000.0 / w, b[1] * 1000.0 / h, b[2] * 1000.0 / w, b[3] * 1000.0 / h]


def build_sref_map(dataset):
    """key (file_name, action, object_category) -> {s_ref, num_pairs}."""
    ann = json.load(open(ANN[dataset]))
    out, miss_prop, dup = {}, 0, 0
    for s in ann:
        fn, action, obj = s["file_name"], s["action"], s["object_category"]
        w, h, boxes, inds = s["width"], s["height"], s["boxes"], s["gt_box_inds"]
        num_pairs = int(s["num_pairs"])
        # pair-ordered boxes_1000 = [p0,o0,p1,o1,...] (matches anchor builder 2i/2i+1)
        b1000 = []
        for i in range(num_pairs):
            b1000.append(norm_box(boxes[inds[2 * i]], w, h))
            b1000.append(norm_box(boxes[inds[2 * i + 1]], w, h))
        stem = os.path.splitext(fn)[0]
        ppath = os.path.join(PROPOSALS_DIR, stem + ".json")
        if os.path.exists(ppath):
            props = json.load(open(ppath)).get("proposals", [])
            props = [{"bbox_2d": p["bbox_1000"], "label": p.get("class_name"),
                      "confidence": p.get("confidence")}
                     for p in props if p.get("bbox_1000")]
        else:
            props, miss_prop = [], miss_prop + 1
        anchor = build_proposal_anchor_grounding(props, b1000, num_pairs)
        s_ref = float(compute_sref_grounding(anchor, {"boxes_1000": b1000, "num_pairs": num_pairs}))
        key = (fn, action, obj)
        if key in out:
            dup += 1
        out[key] = {"s_ref": s_ref, "num_pairs": num_pairs}
    print(f"  [{dataset}] ann={len(ann)} sref_keys={len(out)} dup_keys={dup} images_without_proposals={miss_prop}")
    return out


def bucket_of(s_ref):
    if s_ref < 1e-6:
        return "MISS"
    if s_ref < 0.5:
        return "PARTIAL"
    return "EASY"


def ar_for_items(items):
    """items: list of result dicts with matches_per_threshold. Returns AR, AR@0.5, AR@0.75, n, gt."""
    recalls = {}
    total_gt = 0
    for thr in THRESHOLDS:
        tp = fn = 0
        for it in items:
            m = it["matches_per_threshold"]
            cell = m.get(str(thr)) or m.get(f"{thr:.2f}")
            if cell is None:
                continue
            tp += cell.get("matched", 0)
            fn += cell.get("unmatched_gts", 0)
        recalls[thr] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if thr == 0.5:
            total_gt = tp + fn
    ar = sum(recalls.values()) / len(THRESHOLDS)
    return {
        "AR": ar * 100, "AR@0.5": recalls[0.5] * 100, "AR@0.75": recalls[0.75] * 100,
        "n": len(items), "gt": total_gt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/hoi-benchmarks/stratified_grounding_hard_easy.json")
    args = ap.parse_args()

    report = {}
    for dataset in ["hico", "swig"]:
        print(f"\n=== building s_ref map: {dataset} ===")
        sref_map = build_sref_map(dataset)
        # difficulty distribution over the test set
        dist = {"MISS": 0, "PARTIAL": 0, "EASY": 0}
        for v in sref_map.values():
            dist[bucket_of(v["s_ref"])] += 1
        n_all = sum(dist.values())
        print(f"  difficulty dist: MISS={dist['MISS']} PARTIAL={dist['PARTIAL']} EASY={dist['EASY']} "
              f"| HARD(s_ref<0.5)={dist['MISS']+dist['PARTIAL']} ({100*(dist['MISS']+dist['PARTIAL'])/n_all:.1f}%)")

        report[dataset] = {"dist": dist, "models": {}}
        for row in ROW_ORDER:
            path = RESULTS[row][dataset]
            res = json.load(open(path))
            groups = {"ALL": [], "HARD": [], "EASY": [], "MISS": [], "PARTIAL": []}
            unmatched = 0
            for it in res:
                key = (it["file_name"], it["action"], it["object"])
                info = sref_map.get(key)
                if info is None:
                    unmatched += 1
                    continue
                b = bucket_of(info["s_ref"])
                groups["ALL"].append(it)
                groups[b].append(it)
                if b in ("MISS", "PARTIAL"):
                    groups["HARD"].append(it)
            stats = {g: ar_for_items(items) for g, items in groups.items()}
            stats["_unmatched"] = unmatched
            stats["_n_result"] = len(res)
            report[dataset]["models"][row] = stats
            print(f"  {row:8s} matched={len(groups['ALL'])}/{len(res)} unmatched={unmatched}  "
                  f"AR(all)={stats['ALL']['AR']:.2f}")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")

    # pretty comparison tables
    for dataset in ["hico", "swig"]:
        print(f"\n################ {dataset.upper()} grounding — AR (%) by difficulty ################")
        print(f"{'bucket':<10}{'n':>7} | " + " | ".join(f"{ROW_LABEL[r]:>22}" for r in ROW_ORDER)
              + f" | {'Δ GRPO-base':>12} {'Δ GRPO-SFT':>11}")
        for bucket in ["ALL", "EASY", "HARD", "PARTIAL", "MISS"]:
            n = report[dataset]["models"]["grpo"][bucket]["n"]
            cells = []
            for r in ROW_ORDER:
                s = report[dataset]["models"][r][bucket]
                cells.append(f"{s['AR']:6.2f}/{s['AR@0.5']:5.1f}/{s['AR@0.75']:5.1f}")
            d_gb = report[dataset]["models"]["grpo"][bucket]["AR"] - report[dataset]["models"]["base_8B"][bucket]["AR"]
            d_gs = report[dataset]["models"]["grpo"][bucket]["AR"] - report[dataset]["models"]["sft"][bucket]["AR"]
            print(f"{bucket:<10}{n:>7} | " + " | ".join(f"{c:>22}" for c in cells)
                  + f" | {d_gb:>+12.2f} {d_gs:>+11.2f}")
        print("  (cells = AR / AR@0.5 / AR@0.75 ; Δ on AR)")


if __name__ == "__main__":
    main()
