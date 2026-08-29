"""Verify sftgrpo_prompts.py reproduces the GRPO training prompts BYTE-EXACT.
Rebuilds the grounding + referring user messages from the same inputs the parquet
encodes and diffs against the parquet's verbatim text. Exit 0 + 'OK' iff identical.
"""
import sys, json, re, difflib
import pandas as pd
sys.path.insert(0, "/workspace/hoi-benchmarks")
import sftgrpo_prompts as ap

df = pd.read_parquet("/workspace/AdaTooler-V/verltool/data/train_resampled_moderate.parquet")

def msgs(i): return [dict(m) for m in df.iloc[i]["prompt"]]
def text_of(m):
    c = m["content"]
    return c if isinstance(c, str) else "".join(
        x.get("text", "") for x in (dict(z) for z in c) if "text" in x)

def show_diff(a, b, tag):
    d = list(difflib.unified_diff(a.splitlines(), b.splitlines(), "expected(parquet)", "rebuilt", lineterm=""))
    print(f"  {tag}: {'IDENTICAL' if not d else 'DIFF ↓'}")
    for l in d[:40]: print("   ", l[:200])
    return not d

ok = True

# system prompt
gi = next(i for i in range(len(df)) if df["extra_info"].iloc[i].get("task_type") == "grounding")
ok &= show_diff(text_of(msgs(gi)[0]), ap.SYSTEM_PROMPT, "system")

# grounding user: parse the candidate_objects + action/object back out of the parquet text
g_user = text_of([m for m in msgs(gi) if m["role"] == "user"][0])
g_props = json.loads(re.search(r"<candidate_objects>\n(.*?)\n</candidate_objects>", g_user, re.S).group(1))
gm = re.search(r"Locate every person who is (.*?) and the (.*?) they interact with", g_user, re.S)
obj = gm.group(2)
verb_obj = gm.group(1)            # e.g. "clenching fist"
action = verb_obj[: len(verb_obj) - len(obj)].strip()   # strip trailing object -> "clenching"
rebuilt_g = ap.build_grounding_user(action, obj, g_props)
ok &= show_diff(g_user, rebuilt_g, "grounding user")

# referring user
ri = next(i for i in range(len(df)) if df["extra_info"].iloc[i].get("task_type") == "referring")
r_user = text_of([m for m in msgs(ri) if m["role"] == "user"][0])
r_props = json.loads(re.search(r"<proposals>\n(.*?)\n</proposals>", r_user, re.S).group(1))
pbox = json.loads(re.search(r'located at: \*\*(\{"bbox_2d".*?"label": "person"\})\*\*', r_user).group(1))["bbox_2d"]
obox = json.loads(re.search(r'interacting with is located at: \*\*(\{"bbox_2d".*?"label": "object"\})\*\*', r_user).group(1))["bbox_2d"]
rebuilt_r = ap.build_referring_user(pbox, obox, r_props)
ok &= show_diff(r_user, rebuilt_r, "referring user")

print("\nRESULT:", "OK — byte-exact" if ok else "FAILED — see diffs above")
sys.exit(0 if ok else 1)
