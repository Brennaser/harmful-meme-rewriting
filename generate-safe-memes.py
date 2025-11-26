import os
import json

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import textwrap


MODEL_ID = "digiplay/incursiosMemeDiffusion_v1.6"  # or your local "memesdalle_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_memes_model(model_id: str = MODEL_ID):
    print(f"[+] Loading model {model_id} on {DEVICE}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_background_from_safe_image(pipe, safe_image_prompt: str,
                                        num_inference_steps: int = 25,
                                        guidance_scale: float = 7.5):
    """
    Uses Safe_Image description as the prompt for the meme background.
    """
    prompt = (
        f"meme style illustration. {safe_image_prompt} "
        "simple clear scene with room at the bottom for caption text."
    )

    generator = torch.Generator(device=DEVICE)
    with torch.autocast(DEVICE if DEVICE == "cuda" else "cpu"):
        image = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

    return image


def overlay_caption(image: Image.Image,
                    caption: str,
                    font_path: str = "arial.ttf",
                    position: str = "bottom",
                    max_width_ratio: float = 0.9,
                    margin_ratio: float = 0.03) -> Image.Image:
    """
    Draws Safe_Text on the image as classic meme text.
    """
    img = image.convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    font_size = int(H * 0.06)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    max_text_width = int(W * max_width_ratio)
    chars_per_line = max(10, int(max_text_width / (font_size * 0.6)))
    lines = []
    for line in caption.split("\n"):
        lines.extend(textwrap.wrap(line, width=chars_per_line))

    line_heights = []
    line_widths = []
    for line in lines:
        w, h = draw.textsize(line, font=font)
        line_widths.append(w)
        line_heights.append(h)

    text_height = sum(line_heights) + int(font_size * 0.4) * (len(lines) - 1)
    margin = int(H * margin_ratio)

    if position == "top":
        y_start = margin
    else:
        y_start = H - text_height - margin

    padding = int(font_size * 0.35)
    rect_top = max(0, y_start - padding)
    rect_bottom = min(H, y_start + text_height + padding)
    draw.rectangle([(0, rect_top), (W, rect_bottom)], fill=(0, 0, 0))

    y = y_start
    for line, h in zip(lines, line_heights):
        w, _ = draw.textsize(line, font=font)
        x = (W - w) // 2

        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((x + dx, y + dy), line, font=font, fill="black")

        draw.text((x, y), line, font=font, fill="white")
        y += h + int(font_size * 0.4)

    return img


def generate_safe_meme_from_record(pipe,
                                   record: dict,
                                   out_dir: str) -> str:
    """
    record is one json object with at least:
      id
      Safe_Image
      Safe_Text   optional but recommended
    """
    meme_id = str(record.get("id", "no_id"))
    safe_image_prompt = record.get("Safe_Image", "").strip()
    safe_text = record.get("Safe_Text", "").strip()

    if not safe_image_prompt:
        raise ValueError(f"Record {meme_id} has empty Safe_Image")

    background = generate_background_from_safe_image(pipe, safe_image_prompt)

    if safe_text:
        final_image = overlay_caption(background, safe_text, position="bottom")
    else:
        final_image = background

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{meme_id}_safe.png")
    final_image.save(out_path)
    return out_path


def process_jsonl(jsonl_path: str,
                  out_dir: str,
                  model_id: str = MODEL_ID):
    pipe = load_memes_model(model_id=model_id)

    with open(jsonl_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            try:
                out_path = generate_safe_meme_from_record(pipe, record, out_dir)
                print(f"[ok] {record.get('id')} -> {out_path}")
            except Exception as e:
                print(f"[skip] {record.get('id')} due to error: {e}")


if __name__ == "__main__":
    # edit these paths for your project
    INPUT_JSONL = "train200_subset_with_safe.jsonl"
    OUTPUT_DIR = "safe_memes_out"
    process_jsonl(INPUT_JSONL, OUTPUT_DIR)
