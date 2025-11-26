import os
import csv
import textwrap

from PIL import Image, ImageDraw, ImageFont

INPUT_CSV = "memes_gemini_v2.csv"
IMAGE_DIR = "safe_memes_out_v2"          # where your current images live
OUTPUT_DIR = "safe_memes_with_text"      # new folder with caption overlay
FONT_PATH = "Impact.ttf"                 # optional, can be any .ttf or leave as is


def extract_safe_text(raw: str) -> str:
    """
    Take the gemini_rewrite string and return only the Safe_Text part.
    """
    if raw is None:
        return ""

    raw = raw.split("Safe_Image")[0]
    raw = raw.replace("Safe_Text:", "").strip()
    raw = raw.strip().strip('"').strip()
    return raw


def load_font(img_height: int):
    """
    Try to load Impact or fall back to default font.
    Size is based on image height.
    """
    size = max(20, int(img_height * 0.07))
    try:
        return ImageFont.truetype(FONT_PATH, size=size)
    except Exception:
        return ImageFont.load_default()


def overlay_text_with_pillow(img_path: str, caption: str, out_path: str):
    """
    Open an image, overlay caption at the bottom, save to out_path.
    """
    if not caption.strip():
        # if no caption just copy image
        img = Image.open(img_path).convert("RGB")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        img.save(out_path)
        return

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = load_font(img.height)

    # wrap text for long captions
    wrapped = textwrap.fill(caption.strip(), width=30)

    # use textbbox to measure size
    bbox = draw.textbbox((0, 0), wrapped, font=font, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (img.width - text_w) // 2
    y = img.height - text_h - 20

    # black rectangle background
    bar_top = max(0, y - 10)
    draw.rectangle([(0, bar_top), (img.width, img.height)], fill="black")

    # outline
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        draw.text((x + dx, y + dy), wrapped, font=font, fill="black", align="center")

    # main text
    draw.text((x, y), wrapped, font=font, fill="white", align="center")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)


def process_csv_overlay(input_csv: str, image_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            meme_id = row.get("id", "no_id")
            raw_rewrite = row.get("gemini_rewrite", "")
            safe_text = extract_safe_text(raw_rewrite)

            in_path = os.path.join(image_dir, f"{meme_id}_safe.png")
            out_path = os.path.join(output_dir, f"{meme_id}_safe_text.png")

            if not os.path.exists(in_path):
                print(f"[skip] {meme_id} image not found at {in_path}")
                continue

            try:
                overlay_text_with_pillow(in_path, safe_text, out_path)
                print(f"[✓] {meme_id} -> {out_path}")
            except Exception as e:
                print(f"[skip] {meme_id} due to error: {e}")


if __name__ == "__main__":
    process_csv_overlay(INPUT_CSV, IMAGE_DIR, OUTPUT_DIR)
