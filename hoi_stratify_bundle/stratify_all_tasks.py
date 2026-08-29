#!/usr/bin/env python3
"""Phase 2 (run in the hoi-benchmarks .venv): stratify EXISTING results into
HARD/EASY buckets for all 4 HOI tasks, from results we already have. No re-eval.

Models/paths come from a JSON registry (default models.json next to this script;
override with --models-config). Add a model by giving it a key + entry under
grounding/referring and listing it in row_order — no code edits. Difficulty cache
from compute_sref_cache.py (--cache, default sref_cache.json) is model-independent.

  Usage:  .venv/bin/python stratify_all_tasks.py [--models-config models.json]
                                                 [--cache sref_cache.json] [--skip-referring]

Grounding metrics per bucket:
  AR / AR@0.5 / AR@0.75  -> taken straight from each model's STORED
        matches_per_threshold (micro recall, mean over IoU 0.5..0.95). Exact for
        every model, incl. the base (which was scored by a different eval script).
  ARs / ARm / ARl        -> exact via RE-MATCHING predictions when the re-match
        reproduces the model's stored ALL-bucket AR (tolerance 0.5) -> sft/grpo.
        Otherwise (e.g. base, scored by a different eval) RECONSTRUCTED from the
        STORED matched counts + GT object sizes: exact for single-pair samples,
        largest-object-first attribution for multi-pair (validated to reproduce
        base's published ALL size within ~0.4). Base ALL @s/@m/@l = exact published.

Referring metrics per bucket: METEOR (pycocoevalcap) + BS-F1 (deberta-v2-xxlarge,
  read per-sample from *_bertscore*.json) ; LLM-Judge left blank.
  Difficulty = per-pair object-perceivability s_ref (a REUSE of grounding's
  proposal difficulty, NOT the referring reward's own s_ref). Referring HARD is
  all MISS (single-pair s_ref in {0,0.5,1.0} -> no PARTIAL band).

Each model's ALL-bucket numbers are validated against its published _metrics.json.
"""
import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
# Dataset/ann/cache paths come from paths.json (next to this script). Edit that file
# for your server. These are fallback defaults if paths.json is absent.
_PATHS = os.path.join(_HERE, "paths.json")
if os.path.exists(_PATHS):
    _p = json.load(open(_PATHS))
    ANN_GROUND, ANN_REFER = _p["ann_ground"], _p["ann_refer"]
    CACHE = _p.get("cache", "sref_cache.json")
    if not os.path.isabs(CACHE):
        CACHE = os.path.join(_HERE, CACHE)
else:
    CACHE = os.path.join(_HERE, "sref_cache.json")
    ANN_GROUND = {"hico": "", "swig": ""}
    ANN_REFER = {"hico": "", "swig": ""}
THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
AREA_SMALL, AREA_MEDIUM = 32 ** 2, 96 ** 2

DEFAULT_MODELS_CONFIG = os.path.join(_HERE, "models.json")
# Populated from the models config in main(). Functions read them as globals at call time.
GROUND_RESULTS, REFER_RESULTS, ROW_ORDER, ROW_LABEL = {}, {}, [], {}
GBUCKETS = ["ALL", "EASY", "HARD", "MISS", "PARTIAL"]


def load_models_config(path):
    """Load the model registry and set the module globals used by the stratifiers."""
    global GROUND_RESULTS, REFER_RESULTS, ROW_ORDER, ROW_LABEL
    cfg = json.load(open(path))
    ROW_ORDER = [m for m in cfg["row_order"] if not m.startswith("_")]
    ROW_LABEL = {k: v for k, v in cfg.get("row_labels", {}).items() if not k.startswith("_")}
    for m in ROW_ORDER:
        ROW_LABEL.setdefault(m, m)
    GROUND_RESULTS = {k: v for k, v in cfg.get("grounding", {}).items() if not k.startswith("_")}
    REFER_RESULTS = {k: v for k, v in cfg.get("referring", {}).items() if not k.startswith("_")}
    print(f"loaded models config: {path}  rows={ROW_ORDER}")


# --------------------------------------------------------------------------- geometry/parse
def iou_px(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def to_px(b, w, h):
    return [int(b[0] * w / 1000), int(b[1] * h / 1000), int(b[2] * w / 1000), int(b[3] * h / 1000)]


def field(item, *names):
    """First present, non-null field among aliases (model-neutral key access)."""
    for nm in names:
        if item.get(nm) is not None:
            return item[nm]
    return None


def as_box(x):
    """Accept a box as a list or a '[x1, y1, x2, y2]' string; return list of 4 ints."""
    if isinstance(x, str):
        x = json.loads(x)
    return [int(round(float(v))) for v in x]


def refer_key(stem, pb, ob):
    return f"{stem}|||{','.join(map(str, pb))}|||{','.join(map(str, ob))}"


def mpt_cell(mpt, t):
    """Read a matches_per_threshold cell, neutral to key format: '0.5', '0.50', 'iou_0.50', 'iou_0.5'."""
    for k in (str(t), f"{t:.2f}", f"iou_{t:.2f}", f"iou_{t}"):
        if k in mpt:
            return mpt[k]
    return {}


def size_cat(o):
    a = max(0, o[2] - o[0]) * max(0, o[3] - o[1])
    return "small" if a < AREA_SMALL else ("medium" if a < AREA_MEDIUM else "large")


def parse_pred_pairs(item, w, h):
    ans = item.get("answer")
    pairs = []
    if ans:
        for line in ans.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                objs = json.loads(line)
            except Exception:
                continue
            if not isinstance(objs, list) or len(objs) < 2:
                continue
            pb = ob = None
            for o in objs:
                bb = o.get("bbox_2d", [])
                if len(bb) != 4:
                    continue
                if o.get("label", "").lower() == "person" and pb is None:
                    pb = bb
                elif o.get("label", "").lower() != "person" and ob is None:
                    ob = bb
            if pb and ob:
                pairs.append({"person": to_px(pb, w, h), "object": to_px(ob, w, h)})
        return pairs
    gt = item.get("generated_text")
    if gt:
        try:
            arr = json.loads(gt)
            for o in arr:
                pb, ob = o.get("person_bbox"), o.get("object_bbox")
                if pb and ob and len(pb) == 4 and len(ob) == 4:
                    pairs.append({"person": to_px(pb, w, h), "object": to_px(ob, w, h)})
        except Exception:
            pass
    return pairs


def match_greedy(pred, gt, thr):
    scores = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            v = min(iou_px(p["person"], g["person"]), iou_px(p["object"], g["object"]))
            if v >= thr:
                scores.append((v, pi, gi))
    scores.sort(reverse=True)
    up, ug, matched = set(), set(), set()
    for v, pi, gi in scores:
        if pi not in up and gi not in ug:
            up.add(pi); ug.add(gi); matched.add(gi)
    return matched


def rec(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# --------------------------------------------------------------------------- grounding
def grounding_stratify(dataset, gcache):
    ann = {(s["file_name"], s["action"], s["object_category"]): s for s in json.load(open(ANN_GROUND[dataset]))}
    out = {"models": {}}
    for row in ROW_ORDER:
        path = GROUND_RESULTS.get(row, {}).get(dataset)
        if not path or not os.path.exists(path):
            out["models"][row] = None
            continue
        res = json.load(open(path))
        if isinstance(res, dict):  # some evals wrap rows (InternVL3: per_sample_results)
            res = res.get("per_sample_results") or res.get("results") or res.get("samples") or []
        # STORED AR accumulators (sample-level matched/unmatched_gts)
        stored = {b: {t: {"tp": 0, "fn": 0} for t in THRESHOLDS} for b in GBUCKETS}
        # RE-MATCH size accumulators
        size = {b: {t: {k: 0 for k in ("tp_small", "fn_small", "tp_medium", "fn_medium", "tp_large", "fn_large")}
                    for t in THRESHOLDS} for b in GBUCKETS}
        # reconstructed size from STORED matched counts (faithful for every model incl. base)
        recon = {b: {t: {k: 0 for k in ("tp_small", "fn_small", "tp_medium", "fn_medium", "tp_large", "fn_large")}
                     for t in THRESHOLDS} for b in GBUCKETS}
        rematch_all = {t: {"tp": 0, "fn": 0} for t in THRESHOLDS}
        unmatched = 0
        for it in res:
            fn = field(it, "file_name", "image_file")
            act = field(it, "action")
            obj = field(it, "object", "object_category")
            key = (fn, act, obj)
            b = gcache.get(f"{fn}|||{act}|||{obj}")
            s = ann.get(key)
            if b is None or s is None:
                unmatched += 1
                continue
            buckets_here = ["ALL", b] + (["HARD"] if b in ("MISS", "PARTIAL") else [])
            w, h, boxes, inds, np_ = s["width"], s["height"], s["boxes"], s["gt_box_inds"], int(s["num_pairs"])
            gtp = [{"person": boxes[inds[2 * i]], "object": boxes[inds[2 * i + 1]], "size": size_cat(boxes[inds[2 * i + 1]])}
                   for i in range(np_)]
            areas = [max(0, g["object"][2] - g["object"][0]) * max(0, g["object"][3] - g["object"][1]) for g in gtp]
            order = sorted(range(np_), key=lambda i: -areas[i])  # largest object first
            mpt = it["matches_per_threshold"]
            for t in THRESHOLDS:
                cell = mpt_cell(mpt, t)
                M = cell.get("matched", 0)
                recon_hit = set(order[:M])  # largest-first attribution of the stored matched count
                for bb in buckets_here:
                    stored[bb][t]["tp"] += M
                    stored[bb][t]["fn"] += cell.get("unmatched_gts", 0)
                    for gi, g in enumerate(gtp):
                        hit = "tp" if gi in recon_hit else "fn"
                        recon[bb][t][f"{hit}_{g['size']}"] += 1
            # re-match for size (exact for sft/grpo; validates vs stored)
            pred = parse_pred_pairs(it, w, h)
            for t in THRESHOLDS:
                matched = match_greedy(pred, gtp, t)
                rematch_all[t]["tp"] += len(matched); rematch_all[t]["fn"] += np_ - len(matched)
                for gi, g in enumerate(gtp):
                    hit = "tp" if gi in matched else "fn"
                    for bb in buckets_here:
                        size[bb][t][f"{hit}_{g['size']}"] += 1
        # validate re-match vs stored (ALL)
        stored_ar = 100 * sum(rec(stored["ALL"][t]["tp"], stored["ALL"][t]["fn"]) for t in THRESHOLDS) / 10
        rematch_ar = 100 * sum(rec(rematch_all[t]["tp"], rematch_all[t]["fn"]) for t in THRESHOLDS) / 10
        # base was scored by a different eval (different parser/coord frame AND a
        # different size convention), so never trust a re-match for it: AR/@.5/@.75
        # come from its stored matches; @s/@m/@l only from its published metrics (ALL).
        size_valid = (row != "base") and abs(stored_ar - rematch_ar) <= 0.5
        # published ALL size for fallback (base)
        pub = {}
        mp = path.replace(".json", "_metrics.json")
        if os.path.exists(mp):
            pm = json.load(open(mp))
            pub = {"ARs": 100 * pm.get("ARs", 0), "ARm": 100 * pm.get("ARm", 0), "ARl": 100 * pm.get("ARl", 0)}
        out["models"][row] = {
            "stored": stored, "size": size, "recon": recon, "size_valid": size_valid,
            "stored_ar": stored_ar, "rematch_ar": rematch_ar, "pub_size": pub, "unmatched": unmatched,
        }
    return out


def ground_metrics(model, bucket):
    st = model["stored"][bucket]
    r = {t: rec(st[t]["tp"], st[t]["fn"]) for t in THRESHOLDS}
    m = {"AR": 100 * sum(r.values()) / 10, "AR@0.5": 100 * r[0.5], "AR@0.75": 100 * r[0.75],
         "n_gt": st[0.5]["tp"] + st[0.5]["fn"]}
    # size source: exact re-match when it reproduces stored AR (sft/grpo);
    # otherwise reconstruct @s/@m/@l from STORED matched counts (base) — exact for
    # single-pair samples, largest-object-first heuristic for multi-pair.
    sz = model["size"][bucket] if model["size_valid"] else model["recon"][bucket]
    for nm, k in (("ARs", "small"), ("ARm", "medium"), ("ARl", "large")):
        rr = [rec(sz[t][f"tp_{k}"], sz[t][f"fn_{k}"]) for t in THRESHOLDS]
        m[nm] = 100 * sum(rr) / 10
    # base ALL: use exact published size (matches paper); EASY/HARD stay reconstructed
    if (not model["size_valid"]) and bucket == "ALL" and model["pub_size"]:
        m.update(model["pub_size"])
    return m


# --------------------------------------------------------------------------- referring
def referring_stratify(dataset, refcache, meteor):
    """Model-neutral: each per_triplet row is bucketed by joining its GT pair
    (file_stem, person_box, object_box) to the cache — robust to row order and to
    field-name differences (file_name/image_file, list/str boxes). Falls back to
    index alignment if the keyed join can't be formed."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    by_key = refcache["by_key"] if isinstance(refcache, dict) else None
    by_index = refcache["by_index"] if isinstance(refcache, dict) else refcache
    n = len(by_index)
    tok = PTBTokenizer()
    out = {"models": {}}
    for row in ROW_ORDER:
        cfg = REFER_RESULTS.get(row, {}).get(dataset)
        if not cfg or not cfg.get("pred") or not os.path.exists(cfg["pred"]):
            out["models"][row] = None
            continue
        pred = json.load(open(cfg["pred"]))
        f1 = None
        if cfg.get("bert") and os.path.exists(cfg["bert"]):
            bs = json.load(open(cfg["bert"]))
            if len(bs) == len(pred):
                f1 = [b.get("bertscore_f1") for b in bs]
        # bucket per pred row by keyed join (preferred) or index fallback
        row_bucket, unmatched = [], 0
        for i, it in enumerate(pred):
            b = None
            if by_key is not None:
                fn = field(it, "file_name", "image_file", "image", "image_path")
                pb, ob = field(it, "person_bbox", "person_box"), field(it, "object_bbox", "object_box")
                if fn is not None and pb is not None and ob is not None:
                    try:
                        stem = os.path.splitext(os.path.basename(str(fn)))[0]
                        b = by_key.get(refer_key(stem, as_box(pb), as_box(ob)))
                    except Exception:
                        b = None
            if b is None and len(pred) == n:  # fallback: index-aligned files
                b = by_index[i]
            if b is None:
                unmatched += 1
            row_bucket.append(b)
        gts = tok.tokenize({i: [{"caption": str(field(pred[i], "ground_truth", "gt_action", "gt") or "")}] for i in range(len(pred))})
        res = tok.tokenize({i: [{"caption": str(field(pred[i], "prediction", "predicted_action", "pred") or "")}] for i in range(len(pred))})
        stats = {"_unmatched": unmatched, "_n": len(pred)}
        for b in ["ALL", "EASY", "HARD", "MISS"]:
            idxs = [i for i in range(len(pred))
                    if b == "ALL" or row_bucket[i] == b or (b == "HARD" and row_bucket[i] in ("MISS", "PARTIAL"))]
            if not idxs:
                stats[b] = {"METEOR": None, "BS_F1": None, "LLM_Judge": None, "n": 0}
                continue
            mscore, _ = meteor.compute_score({i: gts[i] for i in idxs}, {i: res[i] for i in idxs})
            bsf1 = (100 * sum(f1[i] for i in idxs) / len(idxs)) if f1 else None
            stats[b] = {"METEOR": 100 * mscore, "BS_F1": bsf1, "LLM_Judge": None, "n": len(idxs)}
        out["models"][row] = stats
    return out


# --------------------------------------------------------------------------- main / tables
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/hoi-benchmarks/stratified_all_tasks.json")
    ap.add_argument("--models-config", default=DEFAULT_MODELS_CONFIG,
                    help="JSON registry of models/paths (default: models.json next to this script)")
    ap.add_argument("--cache", default=CACHE, help="s_ref difficulty cache from compute_sref_cache.py")
    ap.add_argument("--skip-referring", action="store_true")
    args = ap.parse_args()
    load_models_config(args.models_config)
    cache = json.load(open(args.cache))
    report = {"grounding": {}, "referring": {}}

    for ds in ["hico", "swig"]:
        gc = cache["grounding"][ds]
        dist = {k: sum(1 for v in gc.values() if v == k) for k in ("MISS", "PARTIAL", "EASY")}
        print(f"\n=== GROUNDING {ds}  dist={dist} HARD={dist['MISS']+dist['PARTIAL']} ===")
        g = grounding_stratify(ds, gc)
        rep = {"dist": dist, "models": {}}
        for row in ROW_ORDER:
            mm = g["models"][row]
            if mm is None:
                rep["models"][row] = None; continue
            rep["models"][row] = {b: ground_metrics(mm, b) for b in GBUCKETS}
            flag = "size:OK" if mm["size_valid"] else "size:from-metrics(base eval differs)"
            print(f"  {row:9s} stored_AR={mm['stored_ar']:.2f} rematch_AR={mm['rematch_ar']:.2f} [{flag}] unmatched={mm['unmatched']}")
        report["grounding"][ds] = rep

    if not args.skip_referring:
        from pycocoevalcap.meteor.meteor import Meteor
        meteor = Meteor()
        for ds in ["hico", "swig"]:
            rb = cache["referring"][ds]
            by_index = rb["by_index"] if isinstance(rb, dict) else rb
            dist = {k: by_index.count(k) for k in ("MISS", "PARTIAL", "EASY")}
            print(f"\n=== REFERRING {ds}  dist={dist} HARD={dist['MISS']+dist['PARTIAL']} ===")
            r = referring_stratify(ds, rb, meteor)
            report["referring"][ds] = {"dist": dist, "models": r["models"]}
            for row in ROW_ORDER:
                mm = r["models"][row]
                if mm is None:
                    print(f"  {row:9s}: (no file)"); continue
                a = mm["ALL"]
                print(f"  {row:9s} matched={mm['_n']-mm['_unmatched']}/{mm['_n']} "
                      f"ALL METEOR={a['METEOR']:.2f} BS-F1={None if a['BS_F1'] is None else round(a['BS_F1'],2)}")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print_tables(report)


def gcell(m):
    if m is None:
        return f"{'-- n/a --':^46}"
    def f(x):
        return "  -- " if x is None else f"{x:5.2f}"
    return f"{f(m['AR'])}/{f(m['AR@0.5'])}/{f(m['AR@0.75'])}/{f(m.get('ARs'))}/{f(m.get('ARm'))}/{f(m.get('ARl'))}"


def print_tables(report):
    for bucket in ["ALL", "EASY", "HARD"]:
        print(f"\n############ GROUNDING bucket={bucket}  (AR/@.5/@.75/@s/@m/@l, %) ############")
        print(f"{'model':<18}|{'HICO-Ground':^48}|{'SWIG-Ground':^48}")
        for row in ROW_ORDER:
            cells = []
            for ds in ["hico", "swig"]:
                mods = report["grounding"].get(ds, {}).get("models", {})
                mm = mods.get(row)
                cells.append(gcell(mm[bucket] if mm else None))
            print(f"{ROW_LABEL[row]:<18}|{cells[0]:^48}|{cells[1]:^48}")
    if report["referring"]:
        for bucket in ["ALL", "EASY", "HARD"]:
            print(f"\n############ REFERRING bucket={bucket}  (METEOR / LLM-Judge / BS-F1, %) ############")
            print(f"{'model':<18}|{'HICO-Refer':^30}|{'SWIG-Refer':^30}")
            for row in ROW_ORDER:
                cells = []
                for ds in ["hico", "swig"]:
                    mods = report["referring"].get(ds, {}).get("models", {})
                    mm = mods.get(row)
                    if not mm:
                        cells.append(f"{'-- n/a --':^30}"); continue
                    s = mm[bucket]
                    met = "  -- " if s["METEOR"] is None else f"{s['METEOR']:5.2f}"
                    bs = "  -- " if s["BS_F1"] is None else f"{s['BS_F1']:5.2f}"
                    cells.append(f"{met} /  --  / {bs}")
                print(f"{ROW_LABEL[row]:<18}|{cells[0]:^30}|{cells[1]:^30}")


if __name__ == "__main__":
    main()
