import os
import csv

import torch
from diffusers import StableDiffusionXLPipeline
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV = "memes_gemini_v2.csv"
OUTPUT_DIR = "safe_memes_sdxl"


# -------------------------------------------------------------------
# MODEL LOADING AND IMAGE GENERATION
# -------------------------------------------------------------------

def load_model(model_id: str = MODEL_ID):
    print(f"[+] Loading model {model_id} on {DEVICE}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        use_safetensors=True,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_background(pipe, safe_image_prompt: str):
    """
    Generate a clean meme style background using SDXL.
    """
    if not safe_image_prompt:
        safe_image_prompt = "a harmless humorous situation"

    prompt = (
        f"clean cartoon illustration of {safe_image_prompt}, "
        "bright colors, simple background, no text"
    )

    negative = "text, words, letters, caption, typography, font, watermark, logo"

    if DEVICE == "cuda":
        autocast_device = "cuda"
    else:
        autocast_device = "cpu"

    with torch.autocast(autocast_device):
        pil_image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=8.5,
            num_inference_steps=36,
            height=768,
            width=768,
        ).images[0]

    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


# -------------------------------------------------------------------
# TEXT CLEANING AND OPENCV OVERLAY
# -------------------------------------------------------------------

def extract_safe_text(raw: str) -> str:
    """
    Take the gemini_rewrite string and return only the Safe_Text part.

    Example input:
      Safe_Text: "Be careful breathing! ..." Safe_Image: A person ...
    Output:
      Be careful breathing! ...
    """
    if raw is None:
        return ""

    raw = raw.split("Safe_Image")[0]
    raw = raw.replace("Safe_Text:", "").strip()
    raw = raw.strip().strip('"').strip()

    return raw


def overlay_caption_opencv(img_bgr: np.ndarray, caption: str) -> np.ndarray:
    """
    Draw a single line caption at the bottom of the image using OpenCV.
    """
    caption = caption.strip()
    if not caption:
        return img_bgr

    h, w, _ = img_bgr.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(1.2, h / 900.0 * 1.6)
    thickness = max(2, int(h / 450.0))

    (text_w, text_h), baseline = cv2.getTextSize(caption, font, scale, thickness)

    bar_top = int(h * 0.8)
    cv2.rectangle(img_bgr, (0, bar_top), (w, h), (0, 0, 0), -1)

    y = bar_top + (h - bar_top - text_h) // 2 + text_h
    x = max(10, (w - text_w) // 2)

    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        cv2.putText(
            img_bgr,
            caption,
            (x + dx, y + dy),
            font,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )

    cv2.putText(
        img_bgr,
        caption,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return img_bgr


# -------------------------------------------------------------------
# MAIN CSV PIPELINE
# -------------------------------------------------------------------

def process_csv(input_csv: str, output_dir: str, add_caption: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    pipe = load_model()

    with open(input_csv, newline="", encoding="utf8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            meme_id = row.get("id", "no_id")

            raw_rewrite = row.get("gemini_rewrite", "")
            safe_text = extract_safe_text(raw_rewrite)

            safe_image_prompt = f"a cartoon character reacting to this idea: {safe_text}"

            try:
                bg_bgr = generate_background(pipe, safe_image_prompt)

                if add_caption:
                    final_bgr = overlay_caption_opencv(bg_bgr, safe_text)
                else:
                    final_bgr = bg_bgr

                out_path = os.path.join(output_dir, f"{meme_id}_safe_sdxl.png")
                cv2.imwrite(out_path, final_bgr)
                print(f"[✓] {meme_id} -> {out_path}")
            except Exception as e:
                print(f"[skip] {meme_id} due to error: {e}")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_DIR, add_caption=True)
