import os
import csv

import torch
from diffusers import StableDiffusionPipeline
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

# You can change this to SDXL if you want:
# MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
MODEL_ID = "digiplay/incursiosMemeDiffusion_v1.6"
#MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV = "memes_gemini_v2.csv"
OUTPUT_DIR = "safe_memes_out"


# -------------------------------------------------------------------
# MODEL LOADING AND IMAGE GENERATION
# -------------------------------------------------------------------

def load_model(model_id: str = MODEL_ID):
    print(f"[+] Loading model {model_id} on {DEVICE}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe


def generate_background(pipe, safe_image_prompt: str):
    """
    Generate a meme style background using the local diffusion model.
    """
    prompt = f"Generate meme of {safe_image_prompt}"

    with torch.autocast(DEVICE):
        pil_image = pipe(
            prompt=prompt,
            negative_prompt="text, words, letters, caption, typography, font, watermark",
            guidance_scale=9,
            num_inference_steps=40,
            height=768,
            width=768,
        ).images[0]

    # Convert PIL (RGB) -> OpenCV (BGR)
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

    # keep only part before Safe_Image if it exists
    raw = raw.split("Safe_Image")[0]

    # remove Safe_Text label
    raw = raw.replace("Safe_Text:", "").strip()

    # strip outer quotes
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

    # font scale and thickness relative to image size
    scale = max(1.0, h / 800.0 * 1.2)
    thickness = max(2, int(h / 400.0))

    # text size
    (text_w, text_h), baseline = cv2.getTextSize(caption, font, scale, thickness)

    # bottom center
    x = (w - text_w) // 2
    y = h - 40  # 40 px above bottom

    # black bar behind text
    bar_top = max(0, y - text_h - 20)
    cv2.rectangle(img_bgr, (0, bar_top), (w, h), (0, 0, 0), -1)

    # outline (black)
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

    # main text (white)
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

            # you can later swap this to a real Safe_Image column if you add one
            safe_image_prompt = "a harmless, humorous situation that matches the rewritten caption"

            try:
                bg_bgr = generate_background(pipe, safe_image_prompt)

                if add_caption:
                    final_bgr = overlay_caption_opencv(bg_bgr, safe_text)
                else:
                    final_bgr = bg_bgr

                out_path = os.path.join(output_dir, f"{meme_id}_safe.png")
                cv2.imwrite(out_path, final_bgr)
                print(f"[✓] {meme_id} -> {out_path}")
            except Exception as e:
                print(f"[skip] {meme_id} due to error: {e}")


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    process_csv(INPUT_CSV, OUTPUT_DIR, add_caption=True)
