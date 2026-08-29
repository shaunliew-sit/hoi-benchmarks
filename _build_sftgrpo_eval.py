"""Build prompt-ALIGNED eval scripts from the existing SFT eval scripts.

Creates new `eval_*_sftgrpo_qwen3vl.py` + `run_*_sftgrpo_eval.sh` that are
byte-for-byte the originals EXCEPT the prompt is swapped for the verbatim
SFTv2/GRPO training prompt (via sftgrpo_prompts.py). Same CLI/commands, so the
ONE aligned script works for BOTH the SFT and GRPO checkpoints.

Changes applied to each python file:
  1. zoom_in observation text  -> "Here is the zoomed-in view of the requested region."
  2. zoom_out -> text-only obs  "Returned to the full original image view." (+ continue)
  3. inject an override block (before __main__) reassigning SYSTEM_PROMPT +
     build_grounding_prompt / build_referring_prompt / format_proposals to the
     aligned (verbatim training) versions. Python resolves these module globals
     at call time, so main() uses the aligned prompts.
Originals are left untouched.
"""
import os
BASE = "/workspace/hoi-benchmarks"

PY_MAP = {
    "eval_hico_ground_sft_qwen3vl.py": "eval_hico_ground_sftgrpo_qwen3vl.py",
    "eval_swig_ground_sft_qwen3vl.py": "eval_swig_ground_sftgrpo_qwen3vl.py",
    "eval_hico_action_referring_sft_qwen3vl.py": "eval_hico_action_referring_sftgrpo_qwen3vl.py",
    "eval_swig_action_referring_sft_qwen3vl.py": "eval_swig_action_referring_sftgrpo_qwen3vl.py",
}

OBS_OLD = "Here is the zoomed view. Continue your analysis."
OBS_NEW = "Here is the zoomed-in view of the requested region."

# The grounding scripts inject this NON-training "max zoom reached" message; replace
# it with the aligned observation so every turn stays in-distribution.
MAXZOOM_OLD = ("You have used the zoom tool many times. Please now provide your final "
               "answer using the <answer>...</answer> tags based on your analysis so far.")

ZOOMOUT_OLD = (
    '            elif tool_name == "zoom_out":\n'
    '                current_image = original_image\n'
    '                tool_calls_log.append({"name": "zoom_out", "turn": turn})\n'
)
ZOOMOUT_NEW = ZOOMOUT_OLD + (
    '                # ALIGNED: zoom_out returns a TEXT-ONLY observation (matches training)\n'
    '                messages.append({"role": "assistant", "content": text})\n'
    '                messages.append({"role": "user", "content": "Returned to the full original image view."})\n'
    '                continue\n'
)

OVERRIDE = '''\
# ============================================================================
# PROMPT ALIGNMENT OVERRIDE (auto-injected by _build_sftgrpo_eval.py)
# Swap the eval's local prompt constructors for the VERBATIM SFTv2/GRPO training
# prompts (byte-exact; see sftgrpo_prompts.py + _verify_sftgrpo_prompts.py).
# Python resolves these module globals at call time, so main() uses the aligned
# system prompt + user-prompt builders. Same prompt for SFT and GRPO checkpoints.
# ============================================================================
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import sftgrpo_prompts as _ap

SYSTEM_PROMPT = _ap.SYSTEM_PROMPT
format_proposals = _ap.format_proposals

def _strip_image_tag(_p):
    # The verbatim training text begins with the "<image>" placeholder, which the
    # chat processor replaces with the image. Here the image is supplied separately
    # as an image_url content part, so drop the literal placeholder (keep the \\n).
    return _p[len("<image>"):] if _p.startswith("<image>") else _p

def build_grounding_prompt(action, object_category, proposals, is_person_person=False):
    return _strip_image_tag(_ap.build_grounding_user(action, object_category, proposals))

def build_referring_prompt(person_bbox_1000, object_bbox_1000, proposals):
    return _strip_image_tag(_ap.build_referring_user(person_bbox_1000, object_bbox_1000, proposals))
# ============================================================================


'''

for src, dst in PY_MAP.items():
    t = open(os.path.join(BASE, src)).read()
    assert OBS_OLD in t, f"obs literal not found in {src}"
    assert ZOOMOUT_OLD in t, f"zoom_out branch not found in {src}"
    assert 'if __name__ == "__main__":' in t, f"__main__ not found in {src}"
    t = t.replace(OBS_OLD, OBS_NEW)
    t = t.replace(MAXZOOM_OLD, OBS_NEW)            # no-op for action scripts
    t = t.replace(ZOOMOUT_OLD, ZOOMOUT_NEW, 1)
    t = t.replace('if __name__ == "__main__":', OVERRIDE + 'if __name__ == "__main__":', 1)
    open(os.path.join(BASE, dst), "w").write(t)
    print(f"py:  {src} -> {dst}")

# ---- shell scripts ----
SH_MAP = {
    "run_hico_ground_sft_eval.sh": "run_hico_ground_sftgrpo_eval.sh",
    "run_swig_ground_sft_eval.sh": "run_swig_ground_sftgrpo_eval.sh",
    "run_hico_action_sft_eval.sh": "run_hico_action_sftgrpo_eval.sh",
    "run_swig_action_sft_eval.sh": "run_swig_action_sftgrpo_eval.sh",
}
PY_RENAME = list(PY_MAP.items())

for src, dst in SH_MAP.items():
    p = os.path.join(BASE, src)
    if not os.path.exists(p):
        print(f"sh:  WARNING {src} missing, skipped")
        continue
    t = open(p).read()
    for oldpy, newpy in PY_RENAME:
        t = t.replace(oldpy, newpy)
    # distinct default OUTPUT_DIR so aligned results don't overwrite old ones
    t = t.replace("results-sft-grpo-qwen3vl-8b-step1000/", "results-sftgrpo/")
    # match training: max_turns = 5 (GRPO HOI used max_turns=5)
    t = t.replace('MAX_TURNS="${MAX_TURNS:-100}"', 'MAX_TURNS="${MAX_TURNS:-5}"')
    # tag the banner
    t = t.replace("(SFT Qwen3VL)", "(SFT/GRPO training-prompt — Qwen3VL)")
    # ensure shebang is line 1 (some originals had a leading-space shebang)
    if t.lstrip().startswith("#!"):
        t = t.lstrip()
    open(os.path.join(BASE, dst), "w").write(t)
    os.chmod(os.path.join(BASE, dst), 0o755)
    print(f"sh:  {src} -> {dst}")

print("\nDONE.")
