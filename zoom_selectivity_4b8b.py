#!/usr/bin/env python3
"""Verify 'selective tool use emerges at both scales (4B & 8B)'.
Zoom rate (>=1 tool_call) per HARD/EASY stratum for grounding (thinking.jsonl,
join file|||action|||object -> sref_cache grounding bucket) and referring
(per_triplet.json, bucket by_index[triplet_id]). Plus AR / accuracy from metrics.
"""
import json, glob, os

ROOT = "/workspace/hoi-benchmarks/results-sftgrpo-saha-v2"
CACHE = json.load(open("/workspace/hoi-benchmarks/sref_cache.json"))

def hard_easy(b):
    return "EASY" if b == "EASY" else ("HARD" if b in ("MISS", "PARTIAL") else None)

def gdir(ds, model, scale):
    suf = "_step1000" if (model == "grpo" and scale == "4b") else ""
    return os.path.join(ROOT, f"{ds}_ground_{model}_{scale}{suf}")

def adir(ds, model, scale):
    suf = "_step1000" if (model == "grpo" and scale == "4b") else ""
    return os.path.join(ROOT, f"{ds}_action_{model}_{scale}{suf}")

def find(d, pat):
    g = glob.glob(os.path.join(d, pat))
    return g[0] if g else None

# ---------------- GROUNDING zoom selectivity ----------------
print("="*100)
print("1) GROUNDING zoom rate by stratum (zoom = >=1 tool_call in thinking.jsonl)")
print("="*100)
ground_rows = {}
for ds in ["hico", "swig"]:
    gcache = CACHE["grounding"][ds]
    for scale in ["4b", "8b"]:
        for model in ["sft", "grpo"]:
            d = gdir(ds, model, scale)
            tj = find(d, "*_thinking.jsonl")
            buckets = {"HARD": [0,0], "EASY": [0,0], "ALL":[0,0]}  # [zoomed, total]
            unmatched = 0
            for line in open(tj):
                it = json.loads(line)
                key = f"{it['file_name']}|||{it['action']}|||{it['object']}"
                b = gcache.get(key)
                he = hard_easy(b)
                if he is None:
                    unmatched += 1
                    continue
                zoom = 1 if len(it.get("tool_calls") or []) >= 1 else 0
                buckets[he][0]+=zoom; buckets[he][1]+=1
                buckets["ALL"][0]+=zoom; buckets["ALL"][1]+=1
            ground_rows[(ds,scale,model)] = (buckets, unmatched)

def rate(z,t): return 100.0*z/t if t else float('nan')
hdr = f"{'task':<6}{'scale':<5}{'model':<6}|{'HARD zoom%':>11}{'(n)':>9}|{'EASY zoom%':>11}{'(n)':>9}|{'gap(H-E)':>9}|{'ALL%':>7}{'unmatched':>10}"
print(hdr); print("-"*len(hdr))
for ds in ["hico","swig"]:
    for scale in ["4b","8b"]:
        for model in ["sft","grpo"]:
            b,un = ground_rows[(ds,scale,model)]
            hz,ht=b["HARD"]; ez,et=b["EASY"]
            hr,er=rate(hz,ht),rate(ez,et)
            print(f"{ds:<6}{scale:<5}{model:<6}|{hr:>10.1f}%{ht:>9}|{er:>10.1f}%{et:>9}|{hr-er:>8.1f}|{rate(*b['ALL']):>6.1f}%{un:>10}")
    print()

# ---------------- REFERRING zoom ----------------
print("="*100)
print("2) REFERRING zoom rate (per_triplet.json tool_calls, bucket by_index[triplet_id])")
print("="*100)
ref_rows={}
for ds in ["hico","swig"]:
    by_index = CACHE["referring"][ds]["by_index"]
    for scale in ["4b","8b"]:
        for model in ["sft","grpo"]:
            d=adir(ds,model,scale)
            pt=find(d,"*_per_triplet.json")
            data=json.load(open(pt))
            buckets={"HARD":[0,0],"EASY":[0,0],"ALL":[0,0]}
            unmatched=0
            for it in data:
                tid=it["triplet_id"]
                b = by_index[tid] if tid < len(by_index) else None
                zoom = 1 if len(it.get("tool_calls") or [])>=1 else 0
                buckets["ALL"][0]+=zoom; buckets["ALL"][1]+=1
                he=hard_easy(b)
                if he is None:
                    unmatched+=1; continue
                buckets[he][0]+=zoom; buckets[he][1]+=1
            ref_rows[(ds,scale,model)]=(buckets,unmatched)
print(hdr); print("-"*len(hdr))
for ds in ["hico","swig"]:
    for scale in ["4b","8b"]:
        for model in ["sft","grpo"]:
            b,un=ref_rows[(ds,scale,model)]
            hz,ht=b["HARD"]; ez,et=b["EASY"]
            hr,er=rate(hz,ht),rate(ez,et)
            print(f"{ds:<6}{scale:<5}{model:<6}|{hr:>10.1f}%{ht:>9}|{er:>10.1f}%{et:>9}|{hr-er:>8.1f}|{rate(*b['ALL']):>6.1f}%{un:>10}")
    print()

# ---------------- ACCURACY ----------------
print("="*100)
print("3) ACCURACY: grounding AR  |  referring exact_match / METEOR / BS-F1")
print("="*100)
def gmetrics(ds,model,scale):
    m=json.load(open(find(gdir(ds,model,scale),"*_metrics.json")))
    return m.get("AR")*100
print("GROUNDING AR (%):")
print(f"{'task':<6}{'4b-SFT':>9}{'4b-GRPO':>9}{'Δ4b':>8} | {'8b-SFT':>9}{'8b-GRPO':>9}{'Δ8b':>8}")
for ds in ["hico","swig"]:
    s4,g4=gmetrics(ds,'sft','4b'),gmetrics(ds,'grpo','4b')
    s8,g8=gmetrics(ds,'sft','8b'),gmetrics(ds,'grpo','8b')
    print(f"{ds:<6}{s4:>9.3f}{g4:>9.3f}{g4-s4:>+8.3f} | {s8:>9.3f}{g8:>9.3f}{g8-s8:>+8.3f}")

def ametrics(ds,model,scale):
    d=adir(ds,model,scale)
    # main caption metrics file = *_metrics.json WITHOUT 'bertscore' in the name
    mainf=[p for p in glob.glob(os.path.join(d,"*_metrics.json")) if "bertscore" not in p]
    m=json.load(open(mainf[0]))
    em=m.get("exact_match"); met=m.get("METEOR")
    bs=find(d,"*bertscore*_metrics.json")
    f1=None
    if bs:
        bm=json.load(open(bs))
        f1=bm.get("bertscore_f1_mean")
    return em,met,f1
print("\nREFERRING (exact_match / METEOR / BS-F1):")
print(f"{'task':<6}{'scale':<5}{'SFT':>26}{'GRPO':>26}")
for ds in ["hico","swig"]:
    for scale in ["4b","8b"]:
        s=ametrics(ds,'sft',scale); g=ametrics(ds,'grpo',scale)
        def fmt(t):
            em,met,f1=t
            return f"em={em*100:.2f} M={met*100:.2f} F1={'NA' if f1 is None else round(f1*100,2)}"
        print(f"{ds:<6}{scale:<5}{fmt(s):>26}  {fmt(g):>26}")
