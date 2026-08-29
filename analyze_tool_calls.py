#!/usr/bin/env python3
"""
Analyze tool call statistics: SFT-only vs SFT+GRPO models across 4 benchmarks.

Usage:
    python analyze_tool_calls.py [--base-dir /workspace/hoi-benchmarks] [--output results.json]
"""

import json
import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# File mapping
# ---------------------------------------------------------------------------

FILE_MAP = {
    "HICO Action": {
        "SFT": "results-sft-qwen3vl-4b-new/hico_action_sft_new/hico_action_sft_results_20260313_042012_per_triplet.json",
        "SFT+GRPO": "results-sft-grpo-new/hico_action_sft_new/hico_action_sft_results_20260310_141500_per_triplet.json",
        "type": "action",
    },
    "HICO Ground": {
        "SFT": "results-sft-qwen3vl-4b-new/hico_ground_sft_new/hico_ground_sft_results_20260313_035524.json",
        "SFT+GRPO": "results-sft-grpo-new/hico_ground_sft_new/hico_ground_sft_results_20260311_025017.json",
        "type": "ground",
    },
    "SWIG Action": {
        "SFT": "results-sft-qwen3vl-4b-new/swig_action_sft_new/swig_action_sft_results_20260313_045453_per_triplet.json",
        "SFT+GRPO": "results-sft-grpo-new/swig_action_sft_new/swig_action_sft_results_20260310_141513_per_triplet.json",
        "type": "action",
    },
    "SWIG Ground": {
        "SFT": "results-sft-qwen3vl-4b-new/swig_ground_sft_new/swig_ground_sft_results_20260313_035817.json",
        "SFT+GRPO": "results-sft-grpo-new/swig_ground_sft_new/swig_ground_sft_results_20260311_092108.json",
        "type": "ground",
    },
}

BENCHMARK_ORDER = ["HICO Action", "HICO Ground", "SWIG Action", "SWIG Ground"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(filepath: Path) -> list[dict]:
    """Load JSON array from file."""
    with open(filepath) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {filepath}, got {type(data)}")
    return data


def extract_tool_call_counts(samples: list[dict]) -> list[int]:
    """Return list of tool call counts (one per sample)."""
    return [len(s.get("tool_calls", [])) for s in samples]


def extract_tool_type_counts(samples: list[dict]) -> dict[str, int]:
    """Count occurrences of each tool name across all samples."""
    counts: Counter = Counter()
    for s in samples:
        for tc in s.get("tool_calls", []):
            counts[tc.get("name", "unknown")] += 1
    return dict(counts)


def extract_max_turns(samples: list[dict]) -> list[int]:
    """Return max turn number per sample (proxy for conversation depth)."""
    result = []
    for s in samples:
        tool_calls = s.get("tool_calls", [])
        if tool_calls:
            max_turn = max(tc.get("turn", 0) for tc in tool_calls)
        else:
            max_turn = 0
        result.append(max_turn)
    return result


def extract_exact_match(samples: list[dict]) -> list[bool] | None:
    """Return exact_match flags if present (action tasks only)."""
    if not samples or "exact_match" not in samples[0]:
        return None
    return [bool(s.get("exact_match", False)) for s in samples]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(counts: list[int]) -> dict:
    n = len(counts)
    if n == 0:
        return {}
    mean = statistics.mean(counts)
    median = statistics.median(counts)
    std = statistics.stdev(counts) if n > 1 else 0.0
    total = sum(counts)
    pct_zero = 100.0 * counts.count(0) / n
    hist = dict(sorted(Counter(counts).items()))
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "min": min(counts),
        "max": max(counts),
        "total_tool_calls": total,
        "pct_zero": pct_zero,
        "histogram": hist,
    }


def compute_comparison(sft: dict, grpo: dict) -> dict:
    delta_mean = grpo["mean"] - sft["mean"]
    pct_change_mean = 100.0 * delta_mean / sft["mean"] if sft["mean"] > 0 else float("nan")
    delta_median = grpo["median"] - sft["median"]
    delta_pct_zero = grpo["pct_zero"] - sft["pct_zero"]
    return {
        "delta_mean": delta_mean,
        "pct_change_mean": pct_change_mean,
        "delta_median": delta_median,
        "delta_pct_zero": delta_pct_zero,
    }


def compute_correctness_by_tool_usage(
    counts: list[int], exact_match: list[bool]
) -> dict:
    """For action tasks: accuracy split by 0 tool calls vs 1+ tool calls."""
    zero_correct = zero_total = nonzero_correct = nonzero_total = 0
    for c, em in zip(counts, exact_match):
        if c == 0:
            zero_total += 1
            if em:
                zero_correct += 1
        else:
            nonzero_total += 1
            if em:
                nonzero_correct += 1
    result = {}
    if zero_total > 0:
        result["acc_zero_calls"] = 100.0 * zero_correct / zero_total
        result["n_zero_calls"] = zero_total
    if nonzero_total > 0:
        result["acc_nonzero_calls"] = 100.0 * nonzero_correct / nonzero_total
        result["n_nonzero_calls"] = nonzero_total
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def fmt_f(v, decimals=2):
    return f"{v:.{decimals}f}"


def fmt_pct(v):
    return f"{v:+.1f}%" if v >= 0 else f"{v:.1f}%"


def print_summary_table(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("  TOOL CALL STATISTICS: SFT vs SFT+GRPO")
    print("=" * 80)

    header = f"{'Benchmark':<14} {'Model':<10} {'N':>6} {'Mean':>7} {'Median':>7} {'Std':>6} {'Min':>4} {'Max':>4} {'%Zero':>7} {'Total':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for bname in BENCHMARK_ORDER:
        bench = all_results[bname]
        for model in ("SFT", "SFT+GRPO"):
            st = bench[model]["stats"]
            print(
                f"{bname:<14} {model:<10} {st['n']:>6} "
                f"{fmt_f(st['mean']):>7} {fmt_f(st['median']):>7} {fmt_f(st['std']):>6} "
                f"{st['min']:>4} {st['max']:>4} {st['pct_zero']:>6.1f}% {st['total_tool_calls']:>7}"
            )
        cmp = bench["comparison"]
        sign = "▼" if cmp["delta_mean"] < 0 else "▲"
        print(
            f"  {sign} GRPO delta:            "
            f"{'':>6} {fmt_f(cmp['delta_mean'], 2):>7} {fmt_f(cmp['delta_median'], 2):>7} "
            f"{'':>6} {'':>4} {'':>4} {fmt_pct(cmp['delta_pct_zero']):>7}   "
            f"({fmt_pct(cmp['pct_change_mean'])} mean calls)"
        )
        print(sep)


def print_type_breakdown(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("  TOOL TYPE BREAKDOWN (zoom_in vs zoom_out)")
    print("=" * 80)
    header = f"{'Benchmark':<14} {'Model':<10} {'zoom_in':>9} {'zoom_out':>9} {'other':>7}"
    print(header)
    print("-" * len(header))
    for bname in BENCHMARK_ORDER:
        bench = all_results[bname]
        for model in ("SFT", "SFT+GRPO"):
            tc = bench[model]["tool_types"]
            zi = tc.get("zoom_in", 0)
            zo = tc.get("zoom_out", 0)
            other = sum(v for k, v in tc.items() if k not in ("zoom_in", "zoom_out"))
            print(f"{bname:<14} {model:<10} {zi:>9} {zo:>9} {other:>7}")
        print()


def print_histograms(all_results: dict) -> None:
    print("\n" + "=" * 80)
    print("  DISTRIBUTION: Tool Calls per Sample")
    print("=" * 80)

    for bname in BENCHMARK_ORDER:
        bench = all_results[bname]
        sft_st = bench["SFT"]["stats"]
        grpo_st = bench["SFT+GRPO"]["stats"]
        sft_n = sft_st["n"]
        grpo_n = grpo_st["n"]

        # Collect all unique bucket keys, group >=5 together
        all_keys = set(sft_st["histogram"]) | set(grpo_st["histogram"])
        max_key = max(all_keys) if all_keys else 0
        display_max = min(max_key, 4)

        print(f"\n{bname}:")
        print(f"  {'#calls':<8} {'SFT (n / %)':<22} {'SFT+GRPO (n / %)':<22}")
        print(f"  {'-'*8} {'-'*22} {'-'*22}")

        buckets = list(range(display_max + 1))
        if max_key > display_max:
            buckets.append(f"{display_max + 1}+")

        for b in buckets:
            if isinstance(b, str):
                # aggregate 5+
                threshold = display_max + 1
                sft_c = sum(v for k, v in sft_st["histogram"].items() if k >= threshold)
                grpo_c = sum(v for k, v in grpo_st["histogram"].items() if k >= threshold)
                label = b
            else:
                sft_c = sft_st["histogram"].get(b, 0)
                grpo_c = grpo_st["histogram"].get(b, 0)
                label = str(b)

            sft_pct = 100.0 * sft_c / sft_n if sft_n > 0 else 0
            grpo_pct = 100.0 * grpo_c / grpo_n if grpo_n > 0 else 0
            print(f"  {label:<8} {sft_c:>5}  ({sft_pct:5.1f}%)        {grpo_c:>5}  ({grpo_pct:5.1f}%)")


def print_correctness_analysis(all_results: dict) -> None:
    has_any = any(
        all_results[b].get("SFT", {}).get("correctness") or
        all_results[b].get("SFT+GRPO", {}).get("correctness")
        for b in BENCHMARK_ORDER
    )
    if not has_any:
        return

    print("\n" + "=" * 80)
    print("  CORRECTNESS vs TOOL USAGE (Action tasks only)")
    print("=" * 80)
    header = f"{'Benchmark':<14} {'Model':<10} {'Acc (0 calls)':>15} {'N':>7} {'Acc (1+ calls)':>15} {'N':>7}"
    print(header)
    print("-" * len(header))

    for bname in BENCHMARK_ORDER:
        bench = all_results[bname]
        for model in ("SFT", "SFT+GRPO"):
            corr = bench[model].get("correctness")
            if not corr:
                continue
            acc0 = f"{corr.get('acc_zero_calls', float('nan')):.1f}%" if "acc_zero_calls" in corr else "  N/A"
            n0 = corr.get("n_zero_calls", "-")
            acc1 = f"{corr.get('acc_nonzero_calls', float('nan')):.1f}%" if "acc_nonzero_calls" in corr else "  N/A"
            n1 = corr.get("n_nonzero_calls", "-")
            print(f"{bname:<14} {model:<10} {acc0:>15} {n0:>7} {acc1:>15} {n1:>7}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze tool call statistics: SFT vs SFT+GRPO")
    parser.add_argument("--base-dir", default="/workspace/hoi-benchmarks",
                        help="Base directory containing result folders")
    parser.add_argument("--output", default=None,
                        help="Optional path to write JSON output")
    args = parser.parse_args()

    base = Path(args.base_dir)

    # Validate all files exist
    print("Validating file paths...")
    missing = []
    for bname, info in FILE_MAP.items():
        for model in ("SFT", "SFT+GRPO"):
            p = base / info[model]
            if not p.exists():
                missing.append(f"  MISSING [{bname}][{model}]: {p}")
            else:
                print(f"  OK [{bname}][{model}]")
    if missing:
        print("\nERROR: Missing files:")
        for m in missing:
            print(m)
        return

    # Process all benchmarks
    all_results: dict = {}
    for bname in BENCHMARK_ORDER:
        info = FILE_MAP[bname]
        all_results[bname] = {}

        for model in ("SFT", "SFT+GRPO"):
            filepath = base / info[model]
            samples = load_samples(filepath)

            counts = extract_tool_call_counts(samples)
            tool_types = extract_tool_type_counts(samples)
            max_turns = extract_max_turns(samples)
            exact_match = extract_exact_match(samples)

            entry = {
                "stats": compute_stats(counts),
                "max_turn_stats": compute_stats(max_turns),
                "tool_types": tool_types,
                "correctness": None,
            }

            if exact_match is not None:
                entry["correctness"] = compute_correctness_by_tool_usage(counts, exact_match)

            all_results[bname][model] = entry

        all_results[bname]["comparison"] = compute_comparison(
            all_results[bname]["SFT"]["stats"],
            all_results[bname]["SFT+GRPO"]["stats"],
        )

    # Print outputs
    print_summary_table(all_results)
    print_type_breakdown(all_results)
    print_histograms(all_results)
    print_correctness_analysis(all_results)

    # Optional JSON output
    if args.output:
        out_path = Path(args.output)
        # Convert histogram keys to strings for JSON serialization
        serializable = {}
        for bname, bench in all_results.items():
            serializable[bname] = {}
            for model in ("SFT", "SFT+GRPO"):
                entry = bench[model].copy()
                entry["stats"] = entry["stats"].copy()
                entry["stats"]["histogram"] = {
                    str(k): v for k, v in entry["stats"]["histogram"].items()
                }
                entry["max_turn_stats"] = entry["max_turn_stats"].copy()
                entry["max_turn_stats"]["histogram"] = {
                    str(k): v for k, v in entry["max_turn_stats"]["histogram"].items()
                }
                serializable[bname][model] = entry
            serializable[bname]["comparison"] = bench["comparison"]

        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nJSON results written to: {out_path}")


if __name__ == "__main__":
    main()
