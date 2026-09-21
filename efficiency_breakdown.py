"""Per-component efficiency breakdown for SAHA inference (ACCV rebuttal).

Reviewer asked for: detector time, preprocessing and crop overhead, average and
tail latency, tool-call counts, and total cost per image.

Unlike the eval scripts, this runs the object detector LIVE per image rather than
reading the precomputed proposal cache, so the detector cost is actually measured.
Detector config is copied from the script that BUILT that cache
(hoi-dataset-curation/scripts/generate_proposals.py): YOLO on yoloe-26l-seg-pf.pt,
conf=0.3, first-50 detections UNSORTED, ultralytics default imgsz. Verified to
reproduce the cached proposals for a sample image (same count, classes, scores).

Prompt construction, tool parsing and cropping are imported from the real eval
modules so this measures the shipped pipeline, not a reimplementation of it.

Run single-stream on an otherwise idle GPU — concurrent load makes tail latency
a queueing artefact rather than a property of the system.

    python efficiency_breakdown.py --task hico_ground --vllm-url http://vllm-saha:8000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from PIL import Image

YOLOE_WEIGHTS = "/workspace/hoi/saha-hoi/yoloe-26l-seg-pf.pt"
YOLOE_CONF = 0.3
YOLOE_MAX_PROPOSALS = 50
WARMUP_SAMPLES = 3  # discarded: first CUDA calls include lazy init (~1.4s vs ~0.05s)

TASKS = {
    "hico_ground": ("eval_hico_ground_sftgrpo_qwen3vl", "ground",
                    "/workspace/hoi/data/hico_20160224_det/images/test2015"),
    "swig_ground": ("eval_swig_ground_sftgrpo_qwen3vl", "ground",
                    "/workspace/hoi/data/swig_hoi/images_512"),
    "hico_referring": ("eval_hico_action_referring_sftgrpo_qwen3vl", "referring",
                       "/workspace/hoi/data/hico_20160224_det/images/test2015"),
    "swig_referring": ("eval_swig_action_referring_sftgrpo_qwen3vl", "referring",
                       "/workspace/hoi/data/swig_hoi/images_512"),
}


class GpuSampler:
    """Poll nvidia-smi in a sidecar thread to estimate GPU-seconds.

    Only meaningful when this process is the sole GPU consumer; the run script
    checks that before starting.
    """

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                self._samples.append(float(out.stdout.strip().split("\n")[0]))
            except Exception:
                pass  # a dropped sample must not kill the measurement run
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self._samples:
            return {"mean_util_pct": 0.0, "n_samples": 0}
        return {"mean_util_pct": statistics.mean(self._samples),
                "n_samples": len(self._samples)}


def load_detector():
    from ultralytics import YOLO
    return YOLO(YOLOE_WEIGHTS)


def detect_live(model, image_path: str) -> list[dict[str, Any]]:
    """Run the detector and format proposals exactly as the cache generator did."""
    results = model(image_path, conf=YOLOE_CONF, verbose=False)
    r = results[0]
    names, boxes = r.names, r.boxes
    w, h = r.orig_shape[1], r.orig_shape[0]
    out = []
    for i in range(min(len(boxes), YOLOE_MAX_PROPOSALS)):  # first-N, unsorted
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        out.append({
            "class_name": names[int(boxes.cls[i])],
            "bbox": [x1, y1, x2, y2],
            "bbox_1000": [round(x1 * 1000 / w), round(y1 * 1000 / h),
                          round(x2 * 1000 / w), round(y2 * 1000 / h)],
            "confidence": float(boxes.conf[i]),
        })
    return out


def timed_agent_loop(client, model_name: str, messages: list, image: Image.Image,
                     mod, max_turns: int, baseline: bool = False) -> dict[str, Any]:
    """Re-implementation of the eval agent loop with per-turn instrumentation.

    Parsing, cropping and encoding all call the eval module's own helpers, so the
    only thing added here is measurement.
    """
    current = image
    turns: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    crop_s = 0.0
    answer = None

    for turn in range(max_turns):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=model_name, messages=messages, max_tokens=2048, temperature=0.0)
        api_s = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        turns.append({
            "turn": turn, "api_s": api_s,
            "prompt_tokens": getattr(u, "prompt_tokens", None),
            "completion_tokens": getattr(u, "completion_tokens", None),
        })

        answer = mod.extract_answer(text)
        if answer is not None:
            break
        if baseline:
            # Single-pass baseline: one forward pass, no tool loop. A tool request
            # is neither executed nor refused — we simply stop, so this measures the
            # cost of one turn with no agent loop on top.
            break
        tool = mod.parse_tool_call(text)
        if not tool:
            break
        name = tool.get("name", "")
        args = tool.get("arguments", {})
        tc = time.perf_counter()
        if name == "zoom_in":
            bbox = args.get("bbox_2d", [0, 0, 1000, 1000])
            current = mod.execute_zoom_in(image, bbox)
            tool_calls.append({"name": "zoom_in", "turn": turn})
        elif name == "zoom_out":
            current = image
            tool_calls.append({"name": "zoom_out", "turn": turn})
        b64 = mod.image_to_base64(current)
        crop_s += time.perf_counter() - tc

        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64}},
            {"type": "text", "text": "Here is the zoomed view. Continue your analysis."}]})

    return {"turns": turns, "tool_calls": tool_calls, "crop_s": crop_s,
            "answered": answer is not None}


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = min(len(s) - 1, max(0, int(round(q / 100.0 * (len(s) - 1)))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--vllm-url", default="http://vllm-saha:8000")
    ap.add_argument("--model-name", default=None, help="default: auto-detect from server")
    ap.add_argument("--subset", default=None, help="default: subset_annotations/<task>_eff500_seed42.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="debug: only N samples")
    ap.add_argument("--mode", choices=("saha", "baseline"), default="saha",
                    help="saha = full agent loop; baseline = single pass, no tool loop")
    args = ap.parse_args()

    mod_name, kind, img_prefix = TASKS[args.task]
    mod = __import__(mod_name)
    subset = args.subset or f"subset_annotations/{args.task}_eff500_seed42.json"
    suffix = "" if args.mode == "saha" else "_baseline"
    out_path = args.out or f"results-efficiency/{args.task}{suffix}_efficiency.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="x")
    model_name = args.model_name or client.models.list().data[0].id
    rows = json.load(open(subset))
    if args.limit:
        rows = rows[: args.limit]

    print(f"task={args.task}  mode={args.mode}  n={len(rows)}  model={model_name}  kind={kind}")
    print(f"loading detector: {YOLOE_WEIGHTS}")
    t0 = time.perf_counter()
    detector = load_detector()
    print(f"  loaded in {time.perf_counter()-t0:.2f}s")

    sampler = GpuSampler()
    per_sample: list[dict[str, Any]] = []
    run_t0 = time.perf_counter()

    for i, s in enumerate(rows):
        if i == WARMUP_SAMPLES:
            sampler.start()  # start measuring only after CUDA warmup
            run_t0 = time.perf_counter()
        img_path = os.path.join(img_prefix, s["file_name"])

        t = time.perf_counter(); props = detect_live(detector, img_path); det_s = time.perf_counter() - t
        t = time.perf_counter(); image = Image.open(img_path).convert("RGB"); load_s = time.perf_counter() - t
        t = time.perf_counter(); _ = mod.image_to_base64(image); enc_s = time.perf_counter() - t

        t = time.perf_counter()
        if kind == "ground":
            prompt = mod.build_grounding_prompt(s["action"], s["object_category"], props)
        else:
            pb = mod.convert_bbox_to_1000(s["boxes"][s["person_box_idx"]], s["width"], s["height"])
            ob = mod.convert_bbox_to_1000(s["boxes"][s["object_box_idx"]], s["width"], s["height"])
            prompt = mod.build_referring_prompt(pb, ob, props)
        prompt_s = time.perf_counter() - t

        messages = [{"role": "system", "content": mod.SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": mod.image_to_base64(image)}},
                        {"type": "text", "text": prompt}]}]

        t = time.perf_counter()
        loop = timed_agent_loop(client, model_name, messages, image, mod, args.max_turns,
                                baseline=(args.mode == "baseline"))
        vlm_s = time.perf_counter() - t

        per_sample.append({
            "file_name": s["file_name"], "n_proposals": len(props),
            "detector_s": det_s, "image_load_s": load_s, "image_encode_s": enc_s,
            "prompt_build_s": prompt_s, "vlm_total_s": vlm_s,
            "crop_s": loop["crop_s"],
            "vlm_api_s": sum(t["api_s"] for t in loop["turns"]),
            "n_turns": len(loop["turns"]),
            "n_tool_calls": len(loop["tool_calls"]),
            "n_zoom_in": sum(1 for c in loop["tool_calls"] if c["name"] == "zoom_in"),
            "prompt_tokens": sum(t["prompt_tokens"] or 0 for t in loop["turns"]),
            "completion_tokens": sum(t["completion_tokens"] or 0 for t in loop["turns"]),
            "answered": loop["answered"],
            "end_to_end_s": det_s + load_s + enc_s + prompt_s + vlm_s,
            "is_warmup": i < WARMUP_SAMPLES,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(rows)}")

    wall = time.perf_counter() - run_t0
    gpu = sampler.stop()
    scored = [r for r in per_sample if not r["is_warmup"]]

    def stat(field: str) -> dict[str, float]:
        xs = [r[field] for r in scored]
        return {"mean": statistics.mean(xs), "p50": percentile(xs, 50),
                "p95": percentile(xs, 95), "max": max(xs)}

    summary = {
        "task": args.task, "mode": args.mode, "model": model_name, "n": len(scored),
        "n_warmup_discarded": WARMUP_SAMPLES,
        "wall_clock_s": wall,
        "gpu": {**gpu, "gpu_seconds": wall * gpu["mean_util_pct"] / 100.0},
        "components": {f: stat(f) for f in
                       ("detector_s", "image_load_s", "image_encode_s", "prompt_build_s",
                        "crop_s", "vlm_api_s", "vlm_total_s", "end_to_end_s")},
        "counts": {
            "mean_turns": statistics.mean([r["n_turns"] for r in scored]),
            "mean_tool_calls": statistics.mean([r["n_tool_calls"] for r in scored]),
            "zoom_rate": sum(1 for r in scored if r["n_zoom_in"] > 0) / len(scored),
            "mean_prompt_tokens": statistics.mean([r["prompt_tokens"] for r in scored]),
            "mean_completion_tokens": statistics.mean([r["completion_tokens"] for r in scored]),
            "mean_proposals": statistics.mean([r["n_proposals"] for r in scored]),
            "answered_rate": sum(1 for r in scored if r["answered"]) / len(scored),
        },
        "detector_config": {"weights": YOLOE_WEIGHTS, "conf": YOLOE_CONF,
                            "max_proposals": YOLOE_MAX_PROPOSALS,
                            "sorted_by_confidence": False, "imgsz": "ultralytics default"},
    }
    json.dump({"summary": summary, "per_sample": per_sample}, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=2)[:1200])


if __name__ == "__main__":
    main()
