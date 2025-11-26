import os
import csv
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import textwrap


MODEL_ID = "digiplay/incursiosMemeDiffusion_v1.6"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id):
    print(f"[+] Loading model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_background(pipe, safe_image_prompt):
    full_prompt = (
        f"clean cartoon illustration of {safe_image_prompt}, "
        "simple meme background, bright colors, no text, no letters, no words"
    )

    negative = "text, caption, words, letters, font, typography, symbols"

    with torch.autocast(DEVICE):
        return pipe(
            prompt=full_prompt,
            negative_prompt=negative,
            guidance_scale=7.5,
            num_inference_steps=30,
            height=768,
            width=768,
        ).images[0]


def overlay_caption(img, caption):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_size = int(H * 0.06)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # wrap text
    lines = textwrap.wrap(caption, width=40)

    # compute text height using font.getbbox()
    line_heights = []
    for line in lines:
        bbox = font.getbbox(line)  # returns (x0, y0, x1, y1)
        line_height = bbox[3] - bbox[1]
        line_heights.append(line_height)

    text_height = sum(line_heights) + 10
    y_start = H - text_height - 20

    # dark bar
    draw.rectangle([(0, y_start - 10), (W, H)], fill="black")

    # draw text
    y = y_start
    for line, lh in zip(lines, line_heights):
        bbox = font.getbbox(line)
        line_width = bbox[2] - bbox[0]
        x = (W - line_width) // 2

        # outline 4 directions
        for dx, dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            draw.text((x+dx, y+dy), line, font=font, fill="black")

        # main white text
        draw.text((x, y), line, font=font, fill="white")
        y += lh

    return img


def process_csv(csv_path, output_dir, model_id=MODEL_ID):
    os.makedirs(output_dir, exist_ok=True)
    pipe = load_model(model_id)

    with open(csv_path, newline='', encoding="utf8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            meme_id = row["id"]
            safe_text = row["gemini_rewrite"]
            safe_text = safe_text.replace("Safe_Text:", "").strip()


            # universal safe image prompt
            safe_image = "funny meme background illustration"

            bg = generate_background(pipe, safe_image)
            final_img = overlay_caption(bg, safe_text)

            out_path = os.path.join(output_dir, f"{meme_id}_safe.png")
            final_img.save(out_path)

            print(f"[✓] Generated meme for ID {meme_id} → {out_path}")


if __name__ == "__main__":
    INPUT_CSV = "memes_gemini_v2.csv"
    OUTPUT_DIR = "safe_memes_out"
    process_csv(INPUT_CSV, OUTPUT_DIR)
