"""
Zoom-policy ablation harness for SAHA evaluation.

Reviewer-requested ablation: run the same checkpoint on the same benchmark under
four zoom policies, so that the contribution of *adaptive* zooming can be
separated from the contribution of *having a zoom at all*.

    adaptive (default)  the model decides when and where to zoom  -> SAHA as-is
    never               every zoom_in request is denied           -> "no tool" lower bound
    always              the model must zoom at least once before
                        its answer is accepted; it still picks the
                        region                                    -> ablates the *gate* (when)
    random              at least one zoom, but every zoom region
                        is replaced by a random crop              -> ablates the *policy* (where)

Interpretation of the resulting 2x2:
    always  >= adaptive  ->  the gate is not buying anything; just always zoom
    adaptive >  always   ->  selective zooming (knowing when to look) matters
    random  ~= always    ->  gains come from extra pixels/tokens, not from targeting
    always  >  random    ->  the learned region choice (knowing where to look) matters

The policy is enforced in the agent loop, not in the system prompt, so the model
sees exactly the same prompt distribution it was trained on in every condition
(`--never-zoom-prompt-hint` opts into prompt-level enforcement as well).

Usage in an eval script:

    from zoom_ablation import ZoomAblationPolicy, add_zoom_ablation_args

    add_zoom_ablation_args(parser)
    policy = ZoomAblationPolicy.from_args(args)
    ...
    policy.new_episode(idx)                       # per-sample deterministic RNG
    answer, tool_calls, thinking, crops = run_sft_agent_loop(
        ..., zoom_policy=policy, fallback_zoom_bbox=union_box_1000)
    episode = policy.episode_stats()              # per-sample counters
    policy.accumulate(episode)                    # running totals
    ...
    metrics.update(policy.summary(n_samples))     # zoom_rate, forced/denied counts, ...
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ZOOM_MODES = ("adaptive", "never", "always", "random")

# Observation returned to the model when a zoom is refused (never-zoom condition).
DENY_OBSERVATION = (
    "Tool use is disabled for this evaluation. Do not call any tools. "
    "Answer the question directly from the full image and the given proposals, "
    "and put your final answer in an <answer> block."
)

# Observation returned when the model tries to answer before zooming
# (always-zoom / random-zoom conditions).
FORCE_ZOOM_OBSERVATION = (
    "Before you answer, you must inspect the image more closely. "
    "Call the zoom_in tool once on the region that is most relevant to the "
    "question, reason about the zoomed view, and only then give your final "
    "answer in an <answer> block."
)

# Text appended after a deterministic forced crop (model refused to zoom itself).
FORCED_CROP_OBSERVATION = (
    "Here is the zoomed-in view of the requested region. "
    "Now give your final answer in an <answer> block."
)

# Optional prompt-level enforcement for the never-zoom condition.
NEVER_ZOOM_PROMPT_HINT = (
    "\n\n## Tool Availability (this evaluation)\n\n"
    "Visual tools are DISABLED for this run. Do not emit any tool calls. "
    "Reason from the full image and the given proposals only, then answer."
)


@dataclass
class ZoomDecision:
    """What the harness should do with a zoom_in request."""
    action: str              # "allow" | "deny" | "replace"
    bbox: list | None        # bbox actually cropped (1000-normalised), None if denied
    requested_bbox: list | None = None
    note: str = ""           # observation text to send back on "deny"


@dataclass
class ZoomAblationPolicy:
    mode: str = "adaptive"
    seed: int = 0

    # --- always / random: forcing at least one zoom ---------------------------
    max_force_nudges: int = 2       # asks before falling back to a deterministic crop
    fallback_scale: float = 0.7     # centred crop used if the model still refuses

    # --- random: how the replacement region is drawn --------------------------
    random_preserve_size: bool = True   # keep the model's box size, randomise location
    random_min_area: float = 0.04       # else sample area U[min, max] x image area
    random_max_area: float = 0.36
    random_max_iou: float = -1.0        # >=0: reject boxes overlapping the request more than this

    # --- never: prompt-level enforcement on top of harness-level denial -------
    never_prompt_hint: bool = False

    # runtime state ------------------------------------------------------------
    _rng: random.Random = field(default_factory=random.Random, repr=False)
    _episode: dict = field(default_factory=dict, repr=False)
    _totals: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.mode not in ZOOM_MODES:
            raise ValueError(f"Unknown zoom mode {self.mode!r}; expected one of {ZOOM_MODES}")
        self._rng = random.Random(self.seed)
        self._episode = _new_counter()
        self._totals = _new_counter()
        self._totals["samples"] = 0
        self._totals["samples_with_zoom"] = 0

    # -- construction ----------------------------------------------------------
    @classmethod
    def from_args(cls, args) -> "ZoomAblationPolicy":
        return cls(
            mode=getattr(args, "zoom_mode", "adaptive"),
            seed=getattr(args, "zoom_seed", 0),
            max_force_nudges=getattr(args, "force_zoom_nudges", 2),
            fallback_scale=getattr(args, "forced_zoom_scale", 0.7),
            random_preserve_size=not getattr(args, "random_zoom_free_size", False),
            random_min_area=getattr(args, "random_zoom_min_area", 0.04),
            random_max_area=getattr(args, "random_zoom_max_area", 0.36),
            random_max_iou=getattr(args, "random_zoom_max_iou", -1.0),
            never_prompt_hint=getattr(args, "never_zoom_prompt_hint", False),
        )

    # -- properties ------------------------------------------------------------
    @property
    def is_baseline(self) -> bool:
        return self.mode == "adaptive"

    @property
    def requires_zoom(self) -> bool:
        """always/random must contain at least one zoom_in per sample."""
        return self.mode in ("always", "random")

    @property
    def denies_zoom(self) -> bool:
        return self.mode == "never"

    def system_prompt_suffix(self) -> str:
        if self.mode == "never" and self.never_prompt_hint:
            return NEVER_ZOOM_PROMPT_HINT
        return ""

    def apply_to_system_prompt(self, prompt: str) -> str:
        return prompt + self.system_prompt_suffix()

    def nudge_text(self) -> str:
        return FORCE_ZOOM_OBSERVATION

    # -- per-sample lifecycle --------------------------------------------------
    def new_episode(self, index: int) -> None:
        """Re-seed per sample so a run is reproducible and resume-safe."""
        self._rng = random.Random((self.seed * 1000003 + int(index)) & 0x7FFFFFFF)
        self._episode = _new_counter()

    def episode_stats(self) -> dict:
        return dict(self._episode)

    def accumulate(self, episode: dict | None = None) -> None:
        ep = self._episode if episode is None else episode
        self._totals["samples"] += 1
        if ep.get("zoom_executed", 0) > 0:
            self._totals["samples_with_zoom"] += 1
        for k, v in ep.items():
            if k in ("samples", "samples_with_zoom"):
                continue
            self._totals[k] = self._totals.get(k, 0) + v

    def summary(self, n_samples: int | None = None) -> dict:
        n = n_samples if n_samples is not None else self._totals.get("samples", 0)
        n = max(n, 1)
        t = self._totals
        return {
            "zoom_mode": self.mode,
            "zoom_seed": self.seed,
            "zoom_rate": t.get("samples_with_zoom", 0) / n,
            "zooms_per_sample": t.get("zoom_executed", 0) / n,
            "zoom_executed_total": t.get("zoom_executed", 0),
            "zoom_requested_total": t.get("zoom_requested", 0),
            "zoom_denied_total": t.get("zoom_denied", 0),
            "zoom_randomized_total": t.get("zoom_randomized", 0),
            "zoom_forced_total": t.get("zoom_forced", 0),
            "force_nudges_total": t.get("force_nudges", 0),
            "zoom_out_total": t.get("zoom_out", 0),
        }

    def config(self) -> dict:
        cfg = {"zoom_mode": self.mode, "zoom_seed": self.seed}
        if self.requires_zoom:
            cfg["max_force_nudges"] = self.max_force_nudges
            cfg["forced_zoom_scale"] = self.fallback_scale
        if self.mode == "random":
            cfg["random_preserve_size"] = self.random_preserve_size
            cfg["random_area_range"] = [self.random_min_area, self.random_max_area]
            cfg["random_max_iou"] = self.random_max_iou
        if self.mode == "never":
            cfg["never_prompt_hint"] = self.never_prompt_hint
        return cfg

    # -- hooks called from the agent loop --------------------------------------
    def on_zoom_request(self, requested_bbox, turn: int = 0) -> ZoomDecision:
        bbox = _sanitize(requested_bbox)
        self._episode["zoom_requested"] += 1

        if self.mode == "never":
            self._episode["zoom_denied"] += 1
            return ZoomDecision("deny", None, bbox, DENY_OBSERVATION)

        if self.mode == "random":
            new_bbox = self.random_bbox(reference=bbox)
            self._episode["zoom_randomized"] += 1
            self._episode["zoom_executed"] += 1
            return ZoomDecision("replace", new_bbox, bbox)

        self._episode["zoom_executed"] += 1
        return ZoomDecision("allow", bbox, bbox)

    def on_zoom_out(self) -> None:
        self._episode["zoom_out"] += 1

    def on_answer(self, zoom_count: int, nudges_used: int) -> str:
        """
        Called when the model emits <answer>.
        Returns "accept" | "nudge" | "force".
        """
        if not self.requires_zoom or zoom_count > 0:
            return "accept"
        if nudges_used < self.max_force_nudges:
            self._episode["force_nudges"] += 1
            return "nudge"
        return "force"

    def forced_bbox(self, fallback_bbox=None) -> list:
        """Deterministic crop used when the model refuses to zoom in always/random."""
        self._episode["zoom_forced"] += 1
        self._episode["zoom_executed"] += 1
        if self.mode == "random":
            return self.random_bbox(reference=None)
        if fallback_bbox is not None:
            return _sanitize(fallback_bbox)
        s = max(0.1, min(1.0, self.fallback_scale))
        margin = (1.0 - s) / 2.0 * 1000.0
        return [int(margin), int(margin), int(1000 - margin), int(1000 - margin)]

    # -- random region sampling ------------------------------------------------
    def random_bbox(self, reference=None) -> list:
        for _ in range(50):
            box = self._draw_random_bbox(reference)
            if self.random_max_iou < 0 or reference is None:
                return box
            if _iou(box, reference) <= self.random_max_iou:
                return box
        return box  # give up on the IoU constraint rather than loop forever

    def _draw_random_bbox(self, reference) -> list:
        if self.random_preserve_size and reference is not None:
            w = max(1.0, reference[2] - reference[0])
            h = max(1.0, reference[3] - reference[1])
        else:
            area = self._rng.uniform(self.random_min_area, self.random_max_area) * 1e6
            aspect = self._rng.uniform(0.5, 2.0)          # w/h
            w = min(1000.0, (area * aspect) ** 0.5)
            h = min(1000.0, area / w)
        x1 = self._rng.uniform(0.0, max(0.0, 1000.0 - w))
        y1 = self._rng.uniform(0.0, max(0.0, 1000.0 - h))
        return [int(x1), int(y1), int(min(1000.0, x1 + w)), int(min(1000.0, y1 + h))]


def _new_counter() -> dict:
    return {
        "zoom_requested": 0,
        "zoom_executed": 0,
        "zoom_denied": 0,
        "zoom_randomized": 0,
        "zoom_forced": 0,
        "force_nudges": 0,
        "zoom_out": 0,
    }


def _sanitize(bbox) -> list:
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError, IndexError):
        return [0, 0, 1000, 1000]
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return [0, 0, 1000, 1000]
    return [int(x1), int(y1), int(x2), int(y2)]


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def union_bbox(*boxes) -> list:
    """Union of 1000-normalised boxes; handy as the always-zoom fallback region."""
    boxes = [_sanitize(b) for b in boxes if b is not None]
    if not boxes:
        return [0, 0, 1000, 1000]
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def pad_bbox(bbox, pad: float = 0.15) -> list:
    """Expand a 1000-normalised box by `pad` of its size on each side."""
    x1, y1, x2, y2 = _sanitize(bbox)
    dw, dh = (x2 - x1) * pad, (y2 - y1) * pad
    return _sanitize([x1 - dw, y1 - dh, x2 + dw, y2 + dh])


def add_zoom_ablation_args(parser) -> None:
    g = parser.add_argument_group("zoom ablation (SAHA)")
    g.add_argument("--zoom-mode", type=str, default="adaptive", choices=list(ZOOM_MODES),
                   help="adaptive: model decides (default). never: deny all zooms. "
                        "always: force >=1 model-chosen zoom. random: force >=1 zoom "
                        "into a random region.")
    g.add_argument("--zoom-seed", type=int, default=0,
                   help="RNG seed for --zoom-mode random (per-sample derived, resume-safe).")
    g.add_argument("--force-zoom-nudges", type=int, default=2,
                   help="always/random: retries asking the model to zoom before a "
                        "deterministic crop is injected.")
    g.add_argument("--forced-zoom-scale", type=float, default=0.7,
                   help="always: centred-crop scale used when no fallback region is "
                        "available and the model refuses to zoom.")
    g.add_argument("--random-zoom-free-size", action="store_true",
                   help="random: sample box size too (default keeps the model's "
                        "requested box size and only randomises the location).")
    g.add_argument("--random-zoom-min-area", type=float, default=0.04,
                   help="random + --random-zoom-free-size: min box area as a fraction of the image.")
    g.add_argument("--random-zoom-max-area", type=float, default=0.36,
                   help="random + --random-zoom-free-size: max box area as a fraction of the image.")
    g.add_argument("--random-zoom-max-iou", type=float, default=-1.0,
                   help="random: if >=0, resample until the random box overlaps the "
                        "model's requested box by at most this IoU (-1 = pure random).")
    g.add_argument("--never-zoom-prompt-hint", action="store_true",
                   help="never: also tell the model in the system prompt that tools are "
                        "disabled (default enforces at the harness only, keeping the "
                        "prompt identical across conditions).")
