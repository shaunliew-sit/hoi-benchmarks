"""
Single annotated image: SFT Qwen3VL-4B grounding result + GT overlay
for HICO_test2015_00002294.jpg (holding frisbee).

Uses the same PIL drawing style as visualize_horse_panels.py:
  - Dark title bar (top) with yellow title + blue subtitle
  - Coloured bounding boxes with labelled background chips
  - White connecting line between person & object
  - Dark legend strip (bottom)

Colours are standardised to avoid green boxes on green-grass background.
Font sizes increased for readability.
"""
import json
import os
from PIL import Image, ImageDraw, ImageFont

# ── Paths ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "/workspace/data/hico_20160224_det/images/test2015/HICO_test2015_00002294.jpg"
SFT_RESULTS = "/workspace/hoi-benchmarks/results-sft-qwen3vl-4b/hico_ground_sft/hico_ground_sft_results_20260228_145909.json"
ANNOT_PATH  = "/workspace/Groma/groma_data/benchmarks_simplified/hico_ground_test_simplified.json"
OUT_PATH    = "/workspace/data/hico_20160224_det/visualizations/frisbee2_panels/sft_gt_overlay.jpg"

FNAME = "HICO_test2015_00002294.jpg"

# ── Colours (no green — clashes with grass background) ───────────────────────
GT_COLOR   = (255, 210, 0)    # gold  — GT boxes
SFT_PERSON = (0, 180, 255)    # sky-blue — SFT predicted person
SFT_OBJECT = (255, 90, 90)    # red   — SFT predicted object
LINE_COL   = (255, 255, 255)  # white connecting line
TITLE_BG   = (20, 20, 20)

BOX_W = 3


def get_font(size):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_LG = get_font(22)   # title
FONT_SM = get_font(18)   # box labels
FONT_XS = get_font(14)   # legend


def draw_box(draw, box, color, label=None, font=None, width=BOX_W):
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    if label and font:
        tb = draw.textbbox((x1, y1), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        by1 = max(0, y1 - th - 4)
        draw.rectangle([x1, by1, x1 + tw + 4, by1 + th + 4], fill=color)
        draw.text((x1 + 2, by1 + 2), label, fill=(0, 0, 0) if color == (255, 210, 0) else (255, 255, 255), font=font)


def draw_line_between(draw, b1, b2, color):
    c1 = ((b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2)
    c2 = ((b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2)
    draw.line([c1, c2], fill=color, width=2)


def add_title(img, title, subtitle="", bg_color=TITLE_BG):
    bar_h = 36 if not subtitle else 64
    new = Image.new("RGB", (img.width, img.height + bar_h), bg_color)
    new.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(new)
    draw.rectangle([0, 0, img.width, bar_h], fill=bg_color)
    draw.text((8, 4), title, fill=(255, 220, 60), font=FONT_LG)
    if subtitle:
        draw.text((8, 32), subtitle, fill=(180, 230, 255), font=FONT_SM)
    return new


def add_legend(img, items, bg=(20, 20, 20)):
    bar_h = 32
    new = Image.new("RGB", (img.width, img.height + bar_h), bg)
    new.paste(img, (0, 0))
    draw = ImageDraw.Draw(new)
    draw.rectangle([0, img.height, img.width, img.height + bar_h], fill=bg)
    x, y = 8, img.height + 6
    for color, label in items:
        draw.rectangle([x, y + 2, x + 16, y + 18], fill=color,
                       outline=(200, 200, 200), width=1)
        draw.text((x + 20, y), label, fill=(220, 220, 220), font=FONT_XS)
        x += 20 + len(label) * 8 + 14
    return new


def scale_to_pixel(bbox_1000, width, height):
    x1, y1, x2, y2 = bbox_1000
    return [
        max(0, int(x1 * width / 1000)),
        max(0, int(y1 * height / 1000)),
        min(width  - 1, int(x2 * width  / 1000)),
        min(height - 1, int(y2 * height / 1000)),
    ]


def main():
    orig = Image.open(IMAGE_PATH).convert("RGB")
    W, H = orig.size

    # Load GT
    with open(ANNOT_PATH) as f:
        annots = json.load(f)
    annot = next(e for e in annots
                 if e["file_name"] == FNAME and "holding" in e.get("action", ""))
    gt_boxes   = annot["boxes"]
    gt_inds    = annot["gt_box_inds"]
    gt_person  = gt_boxes[gt_inds[0]]
    gt_frisbee = gt_boxes[gt_inds[1]]
    action     = annot["action"]
    obj_cat    = annot["object_category"]

    # Load SFT prediction
    with open(SFT_RESULTS) as f:
        sft_data = json.load(f)
    entry = next(e for e in sft_data
                 if e["file_name"] == FNAME and "holding" in e.get("action", ""))
    preds = json.loads(entry["answer"])
    persons = [p for p in preds if p.get("label") == "person"]
    objects  = [p for p in preds if p.get("label") != "person"]

    pred_person  = scale_to_pixel(persons[0]["bbox_2d"], W, H) if persons else None
    pred_frisbee = scale_to_pixel(objects[0]["bbox_2d"],  W, H) if objects else None

    matched = entry.get("matches_per_threshold", {}).get("0.5", {}).get("matched", 0)
    n_gt    = entry.get("num_gt_pairs", 1)
    tc_cnt  = len(entry.get("tool_calls", []))

    # Draw on image
    img  = orig.copy()
    draw = ImageDraw.Draw(img)

    # GT boxes (gold)
    draw_box(draw, gt_person,  GT_COLOR, "GT person",    FONT_SM)
    draw_box(draw, gt_frisbee, GT_COLOR, f"GT {obj_cat}", FONT_SM)
    draw_line_between(draw, gt_person, gt_frisbee, GT_COLOR)

    # SFT predicted boxes
    if pred_person:
        draw_box(draw, pred_person,  SFT_PERSON, "SFT person",    FONT_SM)
    if pred_frisbee:
        draw_box(draw, pred_frisbee, SFT_OBJECT, f"SFT {obj_cat}", FONT_SM)
    if pred_person and pred_frisbee:
        draw_line_between(draw, pred_person, pred_frisbee, LINE_COL)

    # Legend
    match_sym = "\u2713" if matched == n_gt else "\u2717"
    img = add_legend(img, [
        (GT_COLOR,   f"GT person + {obj_cat}"),
        (SFT_PERSON, "SFT pred person"),
        (SFT_OBJECT, f"SFT pred {obj_cat}  {match_sym} ({matched}/{n_gt})"),
    ])

    # Title bar
    title    = f"08 \u00b7 SFT Qwen3VL-4B \u2014 Grounding"
    subtitle = f"Ground: {matched}/{n_gt} matched @IoU0.5   Tool calls: {tc_cnt}"
    img = add_title(img, title, subtitle)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    img.save(OUT_PATH, quality=95)
    print(f"Saved: {OUT_PATH}  ({img.width}\u00d7{img.height}px)")


if __name__ == "__main__":
    main()
