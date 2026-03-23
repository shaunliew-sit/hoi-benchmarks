"""
SWIG-HOI Action Referring Evaluation Script for Qwen3VL with Proposals

Evaluates Qwen3VL action prediction using METEOR and CIDEr metrics.
Extends the baseline by prepending object proposals to the prompt,
allowing the model to leverage proposal context without tool use.

Task: Given (person, object) bounding boxes, predict the connecting action.

Key Differences from baseline:
- Loads object proposals from PROPOSALS_DIR and prepends them to the prompt
- No tool use, no multi-turn loop (single inference pass)
- Same metrics: METEOR, CIDEr, BLEU, ROUGE-L
"""

import argparse
import base64
import io
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def format_proposals(proposals: list[dict]) -> str:
    """Format proposal list as JSON string for inclusion in prompt."""
    items = []
    for idx, p in enumerate(proposals):
        items.append({
            "id": idx,
            "bbox_2d": p["bbox_1000"],
            "label": p["class_name"],
            "confidence": round(p["confidence"], 2),
        })
    return json.dumps(items, indent=2)


def load_proposals(image_stem: str, proposals_dir: str) -> list[dict]:
    """Load proposals for an image from the proposals directory."""
    proposal_path = os.path.join(proposals_dir, f"{image_stem}.json")
    if not os.path.exists(proposal_path):
        return []
    with open(proposal_path) as f:
        data = json.load(f)
    return data.get("proposals", [])


def extract_response_from_text(response_text, is_thinking_model=False):
    """Extract response text for thinking and instruct models."""
    response_text = response_text.strip()
    prefixes_to_remove = ["output: ", "Output: ", "ANSWER: ", "Answer: "]

    if not is_thinking_model:
        return "", response_text

    if "</think>" in response_text:
        thinking_match = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
        thinking_content = thinking_match[-1].strip() if thinking_match else ""
        final_answer = response_text.rsplit("</think>", 1)[-1].strip()
    else:
        if "<think>" in response_text and "</think>" in response_text:
            parts = response_text.split("</think>")
            thinking_content = parts[0].split("<think>")[-1].strip()
            final_answer = parts[1].strip() if len(parts) > 1 else ""
        else:
            return "", response_text

    for prefix in prefixes_to_remove:
        if final_answer.startswith(prefix):
            final_answer = final_answer[len(prefix) :].strip()
            break

    return thinking_content, final_answer


def build_action_referring_prompt(object_category=None, is_thinking_model=False):
    """Build prompt for action referring task."""
    if is_thinking_model:
        prompt_text = (
            "Task: Describe the action/interaction between the two objects in the two bounding boxes.\n\n"
            "EXPLAIN in thinking:\n"
            "1. What visual cues indicate each object's identity, pose, and positioning\n"
            "2. How the spatial relationship between the two objects suggests an interaction\n"
            "3. What action verb best describes this interaction\n"
            "4. Why this action is more appropriate than alternatives\n\n"
            "IMPORTANT - Use this exact format:\n"
            "<think>\n"
            "Step 1 - Analyze the two objects:\n"
            "[Your detailed reasoning about object identities and poses]\n\n"
            "Step 2 - Analyze the interaction:\n"
            "[Your reasoning about spatial relationships and interaction]\n\n"
            "Step 3 - Determine the action:\n"
            "[Your reasoning about the specific action verb]\n"
            "</think>\n\n"
            "Then output ONLY a SHORT action phrase (2-4 words) in format: [action] [object]\n"
            "Examples: 'riding bicycle', 'holding cup', 'sitting on chair'\n\n"
            "Output Format:\n"
            "Provide ONLY the action phrase (no explanations, no punctuation)."
        )
    else:
        if object_category:
            prompt_text = (
                f"Task: Describe what action the person (in the first bounding box) is performing with the {object_category} (in the second bounding box).\n\n"
                f"Instructions:\n"
                f"1. Focus on the interaction between the person and the {object_category}\n"
                f"2. Provide a SHORT action phrase (2-4 words)\n"
                f"3. Use format: [action] [object] (e.g., 'riding bicycle', 'holding cup')\n\n"
                f"Output Format:\n"
                f"Provide ONLY the action phrase (no explanations, no punctuation)."
            )
        else:
            prompt_text = (
                "Task: Describe what action the person (in the first bounding box) is performing with the object (in the second bounding box).\n\n"
                "Instructions:\n"
                "1. Focus on the interaction between the person and the object\n"
                "2. Provide a SHORT action phrase (2-4 words)\n"
                "3. Include both the action verb and the object\n\n"
                "Output Format:\n"
                "Provide ONLY the action phrase (no explanations, no punctuation)."
            )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image_placeholder"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def convert_bbox_to_qwen_format(bbox, img_size):
    """Convert bounding box to Qwen3VL format [0, 1000]."""
    width, height = img_size
    x1, y1, x2, y2 = bbox
    return [
        int((x1 / width) * 1000),
        int((y1 / height) * 1000),
        int((x2 / width) * 1000),
        int((y2 / height) * 1000),
    ]


def image_to_base64(image):
    """Convert a PIL image to a JPEG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def get_response_text(response):
    """Normalize OpenAI client content into plain text."""
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    return str(content)


def run_inference(
    client,
    model_name,
    image_path,
    person_bbox,
    object_bbox,
    object_category=None,
    is_thinking_model=False,
    proposals=None,
):
    """Run Qwen3VL inference for action referring through a vLLM endpoint."""
    image = Image.open(image_path).convert("RGB")
    img_width, img_height = image.size

    person_bbox_qwen = convert_bbox_to_qwen_format(person_bbox, (img_width, img_height))
    object_bbox_qwen = convert_bbox_to_qwen_format(object_bbox, (img_width, img_height))

    messages = build_action_referring_prompt(object_category, is_thinking_model)
    text_content = messages[0]["content"][1]["text"]

    if is_thinking_model:
        bbox_prompt = (
            f"\n\nFirst bounding box: {person_bbox_qwen}\n"
            f"Second bounding box: {object_bbox_qwen}\n\n"
            f"{text_content}"
        )
    else:
        bbox_prompt = (
            f"\n\nPerson bounding box: {person_bbox_qwen}\n"
            f"Object bounding box: {object_bbox_qwen}\n\n"
            f"{text_content}"
        )

    if proposals:
        proposals_text = format_proposals(proposals)
        proposals_section = (
            "Here are candidate object proposals detected in the image:\n\n"
            f"<proposals>\n{proposals_text}\n</proposals>\n\n"
        )
        bbox_prompt = proposals_section + bbox_prompt

    request_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_base64(image)}},
                {"type": "text", "text": bbox_prompt},
            ],
        }
    ]

    request_kwargs = {
        "model": model_name,
        "messages": request_messages,
    }
    if is_thinking_model:
        request_kwargs.update(
            {
                "max_tokens": 2048,
                "temperature": 0.2,
                "top_p": 0.9,
                "extra_body": {"top_k": 20},
            }
        )
    else:
        request_kwargs.update({"max_tokens": 30, "temperature": 0.0})

    response = client.chat.completions.create(**request_kwargs)
    raw_text = get_response_text(response)
    thinking_content, output_text = extract_response_from_text(raw_text, is_thinking_model)
    return thinking_content, output_text, image


def clean_action_response(response_text):
    """Clean action response to extract action phrase."""
    response = response_text.strip()
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
    ]

    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix) :].strip()
            break

    response = response.rstrip(".!?,;:")
    return response.lower().strip()


_PROPOSAL_COLORS = [
    (255, 165, 0),   # orange
    (148, 0, 211),   # violet
    (0, 206, 209),   # dark turquoise
    (255, 20, 147),  # deep pink
    (50, 205, 50),   # lime green
    (255, 215, 0),   # gold
    (0, 191, 255),   # deep sky blue
    (220, 20, 60),   # crimson
    (127, 255, 0),   # chartreuse
    (255, 99, 71),   # tomato
]


def visualize_action_triplet(
    image_path, person_bbox, object_bbox, predicted_action, gt_action, object_category, output_path,
    proposals=None
):
    """Visualize action referring result as a 3-panel image: Query | Proposals | Result."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

    px1, py1, px2, py2 = person_bbox
    ox1, oy1, ox2, oy2 = object_bbox
    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
    object_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)

    # ── Panel 1: Query (person + object bboxes) ──────────────────────────
    query_img = image.copy()
    q_draw = ImageDraw.Draw(query_img)
    q_draw.rectangle([px1, py1, px2, py2], outline="red", width=4)
    q_draw.text((px1, max(0, py1 - 18)), "Person", fill="red", font=font_small)
    q_draw.rectangle([ox1, oy1, ox2, oy2], outline=(0, 120, 255), width=4)
    q_draw.text((ox1, max(0, oy1 - 18)), object_category.capitalize(), fill=(0, 120, 255), font=font_small)
    q_draw.line([person_center, object_center], fill=(0, 200, 0), width=3)

    # ── Panel 2: Proposals ────────────────────────────────────────────────
    prop_img = image.copy()
    p_draw = ImageDraw.Draw(prop_img)
    if proposals:
        capped = proposals[:20]
        for i, prop in enumerate(capped):
            color = _PROPOSAL_COLORS[i % len(_PROPOSAL_COLORS)]
            bx1, by1, bx2, by2 = prop["bbox_1000"]
            bx1 = int(bx1 * img_w / 1000)
            by1 = int(by1 * img_h / 1000)
            bx2 = int(bx2 * img_w / 1000)
            by2 = int(by2 * img_h / 1000)
            p_draw.rectangle([bx1, by1, bx2, by2], outline=color, width=2)
            label = f"{prop['class_name']} {prop['confidence']:.0%}"
            p_draw.text((bx1 + 2, max(0, by1 - 14)), label, fill=color, font=font_small)
        count_text = f"{len(capped)}/{len(proposals)} proposals shown"
        p_draw.text((5, 5), count_text, fill=(255, 255, 255), font=font_small)
    else:
        p_draw.text((10, img_h // 2 - 10), "No proposals", fill=(128, 128, 128), font=font)

    # ── Panel 3: Result (prediction vs GT with match indicator) ──────────
    result_img = query_img.copy()
    r_draw = ImageDraw.Draw(result_img)
    match = predicted_action.lower().strip() == gt_action.lower().strip()
    match_color = (0, 220, 0) if match else (255, 50, 50)
    r_draw.text((10, 10), f"Pred: {predicted_action}", fill=(255, 255, 255), font=font)
    r_draw.text((10, 35), f"GT:   {gt_action}", fill=(255, 220, 0), font=font)
    r_draw.text((10, 60), "MATCH" if match else "MISMATCH", fill=match_color, font=font)

    # ── Compose 3-panel image ─────────────────────────────────────────────
    header_h = 30
    total_w = img_w * 3
    total_h = img_h + header_h
    final = Image.new("RGB", (total_w, total_h), (220, 220, 220))
    hdr_draw = ImageDraw.Draw(final)

    prop_count = len(proposals) if proposals else 0
    titles = ["Query", f"Proposals ({prop_count})", "Result"]
    for col, title in enumerate(titles):
        hdr_draw.text((col * img_w + 10, 7), title, fill=(0, 0, 0), font=font_small)
        if col > 0:
            hdr_draw.line([(col * img_w, 0), (col * img_w, total_h)], fill=(160, 160, 160), width=2)

    final.paste(query_img, (0, header_h))
    final.paste(prop_img, (img_w, header_h))
    final.paste(result_img, (img_w * 2, header_h))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final.save(output_path)


def normalize_sample(sample):
    """Normalize SWIG sample variants to a common structure."""
    if "boxes" in sample and "person_box_idx" in sample and "object_box_idx" in sample:
        return {
            "file_name": sample["file_name"],
            "triplet_id": sample.get("triplet_id"),
            "person_bbox": sample["boxes"][sample["person_box_idx"]],
            "object_bbox": sample["boxes"][sample["object_box_idx"]],
            "object_category": sample.get("object_category"),
            "gt_action": sample["gt_action"],
        }

    return {
        "file_name": sample["file_name"],
        "triplet_id": sample.get("triplet_id"),
        "person_bbox": sample["person_bbox"],
        "object_bbox": sample["object_bbox"],
        "object_category": sample.get("object_category"),
        "gt_action": sample["gt_action"],
    }


def eval_model(args):
    """Main evaluation function."""
    if not args.vllm_url:
        raise ValueError("--vllm-url is required for vLLM endpoint inference")

    print("=" * 80)
    print("SWIG-HOI Action Referring Evaluation (Qwen3VL + Proposals)")
    print("=" * 80)
    print(f"Model:       {args.model_name}")
    print(f"Device:      {args.device}")
    print(f"vLLM URL:    {args.vllm_url}")
    print(f"Images:      {args.img_prefix}")
    print(f"Annotations: {args.ann_file}")
    if args.proposals_dir:
        print(f"Proposals:   {args.proposals_dir}")
    if args.image:
        print(f"Image filter: {args.image}")
    if args.max_images:
        print(f"Max images:  {args.max_images} (DEBUGGING MODE)")
    if args.verbose:
        print("Verbose:     ENABLED (per-triplet results + visualizations)")
    print("=" * 80)
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    use_wandb = WANDB_AVAILABLE and args.wandb
    if use_wandb:
        print("Initializing Weights & Biases...")
        try:
            wandb.login()
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"swig_action_proposals_qwen3vl_{timestamp}",
                config={
                    "model": args.model_name,
                    "device": args.device,
                    "vllm_url": args.vllm_url,
                    "dataset": "SWIG-HOI-Action",
                    "task": "action_referring",
                    "max_images": args.max_images,
                    "timestamp": timestamp,
                    "proposals_dir": args.proposals_dir,
                },
                tags=["swig", "action-referring", "qwen3vl", "person-person", "vllm", "proposals"],
            )
            print("WandB initialized successfully")
            print(f"  Run URL: {wandb.run.url}")
            print(f"  Project: {wandb.run.project}")
            print(f"  Run name: {wandb.run.name}\n")
        except Exception as e:
            print(f"Warning: WandB initialization failed: {e}")
            print("Continuing evaluation without WandB logging...\n")
            use_wandb = False

    client = OpenAI(base_url=f"{args.vllm_url}/v1", api_key="placeholder")
    is_thinking_model = "Thinking" in args.model_name or "thinking" in args.model_name
    print(f"Model type: {'Thinking' if is_thinking_model else 'Instruct'}\n")

    print(f"Loading annotations from: {args.ann_file}")
    with open(args.ann_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "images" in data and "annotations" in data:
        print("Detected COCO format annotation file")
        images_dict = {img["id"]: img for img in data["images"]}
        annotations_dict = {ann["image_id"]: ann for ann in data["annotations"]}
        dataset_samples = []
        for img_id, img_info in images_dict.items():
            ann = annotations_dict.get(img_id)
            if ann:
                dataset_samples.append(
                    {
                        "file_name": img_info["file_name"],
                        "triplet_id": img_id,
                        "person_bbox": img_info["subject_bbox"],
                        "object_bbox": img_info["object_bbox"],
                        "object_category": img_info.get("object_category", "object"),
                        "gt_action": ann["caption"],
                    }
                )
    elif isinstance(data, list):
        print("Detected regular format annotation file")
        dataset_samples = []
        for idx, sample in enumerate(data):
            if "boxes" in sample and "conversation" in sample:
                box_inds = sample["conversation"][0]["box_inds"]
                boxes = sample["boxes"]
                dataset_samples.append(
                    {
                        "file_name": sample["file_name"],
                        "triplet_id": idx,
                        "person_bbox": boxes[box_inds[0]],
                        "object_bbox": boxes[box_inds[1]],
                        "object_category": sample.get("object_category"),
                        "gt_action": sample["conversation"][1]["value"],
                    }
                )
            else:
                dataset_samples.append(normalize_sample(sample))
    else:
        raise ValueError("Unknown annotation format")

    print(f"Loaded {len(dataset_samples)} action referring triplets")

    if args.image:
        dataset_samples = [sample for sample in dataset_samples if args.image in sample["file_name"]]
        print(f"After image filter '{args.image}': {len(dataset_samples)} triplets")

    if args.max_images is not None and args.max_images < len(dataset_samples):
        print(f"\nLimiting evaluation to first {args.max_images} triplets")
        dataset_samples = dataset_samples[: args.max_images]

    print(f"\nDataset: {len(dataset_samples)} triplets")
    print("=" * 80)

    predictions = []
    per_triplet_results = []
    missing_proposals = 0
    action_stats = defaultdict(
        lambda: {
            "total": 0,
            "exact_match": 0,
            "predictions": [],
            "ground_truths": [],
        }
    )

    partial_file = args.pred_file + ".partial.jsonl"
    processed_indices = set()
    if args.resume and os.path.exists(partial_file):
        print(f"\nResuming from partial checkpoint: {partial_file}")
        loaded = 0
        with open(partial_file, "r") as partial_in:
            for line in partial_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    idx = rec["idx"]
                    triplet_result = rec["triplet_result"]
                    prediction = rec["prediction"]
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Warning: skipping corrupt partial record: {e}")
                    continue

                processed_indices.add(idx)
                per_triplet_results.append(triplet_result)
                predictions.append(prediction)
                gt_action = triplet_result["ground_truth"]
                pred_action = triplet_result["prediction"]
                action_stats[gt_action]["total"] += 1
                action_stats[gt_action]["exact_match"] += 1 if triplet_result["exact_match"] else 0
                action_stats[gt_action]["predictions"].append(pred_action)
                action_stats[gt_action]["ground_truths"].append(gt_action)
                loaded += 1

        print(f"Loaded {loaded} completed samples, resuming from next unprocessed...")
    elif args.resume:
        print(f"No partial checkpoint found at {partial_file}, starting fresh")

    partial_f = open(partial_file, "a")

    viz_dir = None
    if args.verbose:
        viz_dir = args.pred_file.replace(".json", "_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        print(f"Visualization directory: {viz_dir}\n")

    show_verbose = args.verbose or len(dataset_samples) <= 100

    print("\nStarting evaluation...")
    for idx, sample in enumerate(tqdm(dataset_samples, disable=show_verbose)):
        if idx in processed_indices:
            continue

        sample = normalize_sample(sample)
        file_name = sample["file_name"]
        img_path = os.path.join(args.img_prefix, file_name)
        person_bbox = sample["person_bbox"]
        object_bbox = sample["object_bbox"]
        gt_action = sample["gt_action"]

        object_category = sample.get("object_category")
        if not object_category or object_category == "object":
            parts = gt_action.split()
            object_category = parts[-1] if len(parts) >= 2 else None

        # Load proposals if proposals_dir is set
        proposals = None
        if args.proposals_dir:
            image_stem = os.path.splitext(file_name)[0]
            proposals = load_proposals(image_stem, args.proposals_dir)
            if not proposals:
                missing_proposals += 1

        if show_verbose:
            print(f"\n[Triplet {idx + 1}/{len(dataset_samples)}] {file_name}")
            print(f"  Person bbox: {person_bbox}")
            print(f"  Object bbox: {object_bbox}")
            print(f"  GT action: {gt_action}")
            if args.proposals_dir:
                print(f"  Proposals: {len(proposals) if proposals else 0} found")

        thinking_content, output_text, _ = run_inference(
            client,
            args.model_name,
            img_path,
            person_bbox,
            object_bbox,
            object_category,
            is_thinking_model,
            proposals=proposals,
        )

        if show_verbose:
            if thinking_content:
                print(f"  Thinking: {thinking_content[:150]}...")
            print(f"  Raw output: {output_text[:100]}...")

        predicted_action = clean_action_response(output_text)

        if show_verbose:
            print(f"  Predicted: {predicted_action}")

        prediction = {"image_id": idx, "caption": predicted_action}
        predictions.append(prediction)

        exact_match = predicted_action.lower().strip() == gt_action.lower().strip()
        triplet_result = {
            "triplet_id": idx,
            "file_name": file_name,
            "person_bbox": person_bbox,
            "object_bbox": object_bbox,
            "ground_truth": gt_action,
            "prediction": predicted_action,
            "raw_output": output_text[:200],
            "exact_match": exact_match,
        }
        if thinking_content:
            triplet_result["thinking_content"] = thinking_content

        per_triplet_results.append(triplet_result)

        action_stats[gt_action]["total"] += 1
        action_stats[gt_action]["exact_match"] += 1 if exact_match else 0
        action_stats[gt_action]["predictions"].append(predicted_action)
        action_stats[gt_action]["ground_truths"].append(gt_action)

        partial_f.write(json.dumps({"idx": idx, "prediction": prediction, "triplet_result": triplet_result}) + "\n")
        partial_f.flush()

        if viz_dir is not None:
            viz_path = os.path.join(viz_dir, f"{idx:05d}_{file_name}")
            try:
                visualize_action_triplet(
                    img_path,
                    person_bbox,
                    object_bbox,
                    predicted_action,
                    gt_action,
                    object_category or "object",
                    viz_path,
                    proposals=proposals,
                )

                if use_wandb:
                    wandb.log(
                        {
                            f"visualization/{idx:04d}": wandb.Image(
                                viz_path,
                                caption=f"{file_name} | Pred: {predicted_action} | GT: {gt_action}",
                            )
                        }
                    )
            except Exception as e:
                if show_verbose:
                    print(f"  Warning: Visualization failed: {e}")

        if use_wandb:
            wandb_metrics = {"sample_idx": idx, "exact_match": 1 if exact_match else 0}
            if thinking_content:
                wandb_metrics["has_thinking"] = 1
            wandb.log(wandb_metrics)

    partial_f.close()

    print("\n" + "=" * 80)
    print("Computing METEOR and CIDEr metrics...")
    print("=" * 80)

    annotations = []
    images_info = []
    for idx, sample in enumerate(dataset_samples):
        sample = normalize_sample(sample)
        images_info.append({"id": idx})
        annotations.append({"image_id": idx, "caption": sample["gt_action"], "id": idx})

    gt_coco_format = {
        "info": {
            "description": "SWIG-HOI Action Referring Ground Truth",
            "version": "1.0",
            "year": 2025,
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": images_info,
        "annotations": annotations,
        "type": "captions",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(gt_coco_format, f)
        gt_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(predictions, f)
        pred_file = f.name

    coco = COCO(gt_file)
    coco_result = coco.loadRes(pred_file)
    coco_eval = COCOEvalCap(coco, coco_result)
    try:
        coco_eval.evaluate()
    except Exception as e:
        print(f"\nWarning: SPICE metric failed: {type(e).__name__}")
        print("Continuing with BLEU, METEOR, ROUGE_L, CIDEr metrics.")

    os.unlink(gt_file)
    os.unlink(pred_file)

    print("\n" + "=" * 80)
    print("SWIG-HOI Action Referring Results (Qwen3VL + Proposals)")
    print("=" * 80)
    print(f"{'Metric':<15} {'Score':>10}  {'Description':<50}")
    print("-" * 80)

    metrics = {}
    for metric, score in coco_eval.eval.items():
        metrics[metric] = score
        desc = {
            "BLEU_1": "BLEU-1 (unigram overlap)",
            "BLEU_2": "BLEU-2 (bigram overlap)",
            "BLEU_3": "BLEU-3 (trigram overlap)",
            "BLEU_4": "BLEU-4 (4-gram overlap)",
            "METEOR": "METEOR (semantic similarity)",
            "ROUGE_L": "ROUGE-L (longest common subsequence)",
            "CIDEr": "CIDEr (corpus consensus)",
            "SPICE": "SPICE (semantic propositional content)",
        }.get(metric, metric)
        print(f"{metric:<15} {score * 100:>9.2f}%  {desc:<50}")

    total_triplets = len(per_triplet_results)
    exact_matches = sum(1 for r in per_triplet_results if r["exact_match"])
    exact_match_acc = exact_matches / total_triplets if total_triplets > 0 else 0.0
    metrics["exact_match"] = exact_match_acc

    print("-" * 80)
    print(f"{'Exact Match':<15} {exact_match_acc * 100:>9.2f}%  {'Exact string match accuracy':<50}")

    if args.proposals_dir:
        metrics["missing_proposals"] = missing_proposals
        print(f"{'Missing proposals':<15} {missing_proposals:>9}    {'Images without proposal files':<50}")

    print("=" * 80)

    if use_wandb:
        wandb.log(metrics)
        wandb.log({"total_triplets": total_triplets, "exact_matches": exact_matches})
        if args.proposals_dir:
            wandb.log({"missing_proposals": missing_proposals})

    print("\n" + "=" * 80)
    print("Per-Action Statistics (Top 20 by frequency)")
    print("=" * 80)
    print(f"{'Action':<30} {'Count':>8} {'Exact Match':>12}")
    print("-" * 80)

    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    for action, stats in sorted_actions[:20]:
        accuracy = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"{action:<30} {stats['total']:>8} {accuracy * 100:>11.1f}%")

    print("=" * 80)

    os.makedirs(os.path.dirname(os.path.abspath(args.pred_file)), exist_ok=True)

    print(f"\nSaving predictions to: {args.pred_file}")
    with open(args.pred_file, "w") as f:
        json.dump(predictions, f, indent=2)

    metrics_file = args.pred_file.replace(".json", "_metrics.json")
    print(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.verbose:
        per_triplet_file = args.pred_file.replace(".json", "_per_triplet.json")
        print(f"Saving per-triplet results to: {per_triplet_file}")
        with open(per_triplet_file, "w") as f:
            json.dump(per_triplet_results, f, indent=2)

        per_action_file = args.pred_file.replace(".json", "_per_action.json")
        print(f"Saving per-action stats to: {per_action_file}")
        action_stats_dict = {action: stats for action, stats in action_stats.items()}
        with open(per_action_file, "w") as f:
            json.dump(action_stats_dict, f, indent=2)

    if is_thinking_model:
        thinking_samples = [s for s in per_triplet_results if s.get("thinking_content")]
        if thinking_samples:
            thinking_file = args.pred_file.replace(".json", "_thinking.jsonl")
            print(f"Saving thinking content to: {thinking_file}")
            with open(thinking_file, "w") as f:
                for result in thinking_samples:
                    thinking_entry = {
                        "file_name": result["file_name"],
                        "triplet_id": result["triplet_id"],
                        "ground_truth": result["ground_truth"],
                        "prediction": result["prediction"],
                        "thinking_content": result["thinking_content"],
                        "raw_output": result["raw_output"],
                    }
                    f.write(json.dumps(thinking_entry) + "\n")
            print(f"  Saved {len(thinking_samples)} samples with thinking content")

    if os.path.exists(partial_file):
        os.remove(partial_file)
        print(f"Removed partial checkpoint: {partial_file}")

    if viz_dir is not None:
        viz_count = len([f for f in os.listdir(viz_dir) if f.endswith(".jpg")])
        print(f"\nVisualizations: {viz_dir}/")
        print(f"  Total images saved: {viz_count}")

    if use_wandb:
        wandb.save(args.pred_file)
        wandb.save(metrics_file)
        if args.verbose:
            wandb.save(per_triplet_file)
            wandb.save(per_action_file)
        if is_thinking_model and thinking_samples:
            wandb.save(thinking_file)
            wandb.log({"thinking_samples_count": len(thinking_samples)})

        action_table_data = []
        for action, stats in sorted_actions[:20]:
            accuracy = stats["exact_match"] / stats["total"] if stats["total"] > 0 else 0.0
            action_table_data.append([action, stats["total"], stats["exact_match"], f"{accuracy:.1%}"])

        wandb.log(
            {
                "action_performance_table": wandb.Table(
                    columns=["Action", "Total", "Exact Match", "Accuracy"],
                    data=action_table_data,
                )
            }
        )

        wandb.finish()
        print("WandB logging complete")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWIG-HOI Action Referring Evaluation with Qwen3VL + Proposals")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Qwen3VL model name")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (kept for compatibility; inference now uses the vLLM endpoint)",
    )
    parser.add_argument("--ann-file", type=str, required=True, help="Path to SWIG action referring annotation file")
    parser.add_argument("--img-prefix", type=str, required=True, help="Path to SWIG images directory (images_512)")
    parser.add_argument("--pred-file", type=str, required=True, help="Output file for predictions")
    parser.add_argument("--vllm-url", type=str, default=None, help="URL of running vLLM server")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Filter samples to those whose file_name contains this string",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Limit evaluation to first N triplets")
    parser.add_argument("--verbose", action="store_true", help="Show detailed per-triplet results and visualizations")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="swig-action-referring-qwen3vl", help="W&B project")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")
    parser.add_argument("--resume", action="store_true", help="Resume from partial checkpoint if available")
    parser.add_argument(
        "--proposals-dir",
        type=str,
        default=None,
        help="Path to object proposals directory (optional; if set, proposals are prepended to prompt)",
    )

    eval_model(parser.parse_args())
