#!/usr/bin/env python3
"""
HOI Multi-Model Comparison Visualization Generator.
Run with: source .venv/bin/activate && python generate_visualizations.py
"""
import json
import re
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.resolve()
PROPOSALS_DIR = Path("/workspace/hoi-tool-use-checkpoints/test_proposals")
HICO_IMAGES = Path("/workspace/data/hico_20160224_det/images/test2015")
SWIG_IMAGES = Path("/workspace/data/swig_hoi/images_512")
ANNOT_DIR = Path("/workspace/Groma/groma_data/benchmarks_simplified")
OUTPUT_DIR = REPO / "visualizations"

# ── Result file paths ────────────────────────────────────────────────────────
RESULTS = {
    "hico_ground": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/hico_ground_qwen3vl_instruct/hico_ground_qwen3vl_results_20260302_132553.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/hico_ground_sft/hico_ground_sft_results_20260226_144052.json",
        "grpo_think":REPO / "results-sft-grpo-step140/hico_ground_sft/hico_ground_sft_results_20260226_144052_thinking.jsonl",
    },
    "hico_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/hico_action_qwen3vl_instruct/hico_action_qwen3vl_results_20260302_132528_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/hico_action_sft/hico_action_sft_results_20260228_145855_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/hico_action_sft/hico_action_sft_results_20260226_144129_per_triplet.json",
        "grpo_think":REPO / "results-sft-grpo-step140/hico_action_sft/hico_action_sft_results_20260226_144129_thinking.jsonl",
    },
    "swig_ground": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_ground_qwen3vl_instruct/swig_ground_qwen3vl_results_20260302_132422.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_ground_sft/swig_ground_sft_results_20260228_145937_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/swig_ground_sft/swig_ground_sft_results_20260226_143835.json",
        "grpo_think":REPO / "results-sft-grpo-step140/swig_ground_sft/swig_ground_sft_results_20260226_143835_thinking.jsonl",
    },
    "swig_refer": {
        "baseline": REPO / "results-baseline-qwen3vl-4b-instruct/swig_action_qwen3vl_instruct/swig_action_qwen3vl_results_20260302_132455_per_triplet.json",
        "sft":      REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_per_triplet.json",
        "sft_think":REPO / "results-sft-qwen3vl-4b/swig_action_sft/swig_action_sft_results_20260228_145929_thinking.jsonl",
        "grpo":     REPO / "results-sft-grpo-step140/swig_action_sft/swig_action_sft_results_20260226_143917_per_triplet.json",
        "grpo_think":REPO / "results-sft-grpo-step140/swig_action_sft/swig_action_sft_results_20260226_143917_thinking.jsonl",
    },
}

ANNOT_FILES = {
    "hico_ground": ANNOT_DIR / "hico_ground_test_simplified.json",
    "hico_refer":  ANNOT_DIR / "hico_action_referring_test_simplified.json",
    "swig_ground": ANNOT_DIR / "swig_ground_test_simplified.json",
    "swig_refer":  ANNOT_DIR / "swig_action_referring_test_simplified.json",
}

# ── Mandatory images (MUST appear in visualizations) ────────────────────────
MANDATORY = {
    "hico_ground": ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "hico_refer":  ["HICO_test2015_00009124", "HICO_test2015_00003584", "HICO_test2015_00006612"],
    "swig_ground": ["checking_24"],
    "swig_refer":  ["checking_24"],
}

# ── Scoring thresholds ───────────────────────────────────────────────────────
WRONG_PROPOSAL_THRESHOLD = 3
BBOX_DIVERGE_IOU_THRESH   = 0.15
ZOOM_DIVERGE_IOU_THRESH   = 0.05
IMAGES_PER_TASK           = 10

# ── Colors ───────────────────────────────────────────────────────────────────
COLORS = {
    "person":   "#FF4444",
    "object":   "#4444FF",
    "gt":       "#22AA22",
    "proposal": "#FF8800",
    "pred":     "#AA22FF",
    "baseline": "#00AAAA",
}


def setup_output_dirs():
    for task in ["hico_ground", "hico_refer", "swig_ground", "swig_refer"]:
        for subdir in ["comparison", "sft_detail", "grpo_detail", "reasoning"]:
            (OUTPUT_DIR / task / subdir).mkdir(parents=True, exist_ok=True)
    print(f"Output dirs created under {OUTPUT_DIR}")


def verify_paths():
    missing = []
    for label, d in [("PROPOSALS_DIR", PROPOSALS_DIR), ("HICO_IMAGES", HICO_IMAGES),
                     ("SWIG_IMAGES", SWIG_IMAGES), ("ANNOT_DIR", ANNOT_DIR)]:
        if not d.exists():
            missing.append(f"  [dir][{label}]: {d}")
    for task, paths in RESULTS.items():
        for key, p in paths.items():
            if p is not None and not Path(p).exists():
                missing.append(f"  [{task}][{key}]: {p}")
    for task, p in ANNOT_FILES.items():
        if not Path(p).exists():
            missing.append(f"  [annot][{task}]: {p}")
    if missing:
        detail = "\n".join(missing)
        raise FileNotFoundError(f"Fix missing files before continuing:\n{detail}")
    print("All source files verified OK")


if __name__ == "__main__":
    setup_output_dirs()
    verify_paths()
    print("Setup complete.")
