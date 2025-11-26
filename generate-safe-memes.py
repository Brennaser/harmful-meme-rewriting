import os
from huggingface_hub import login

token = os.environ.get("HUGGINGFACE_TOKEN")
login(token=token)

import os
import csv

import torch
from diffusers import AutoPipelineForText2Image
import cv2
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CSV = "memes_gemini_v2.csv"
OUTPUT_DIR = "safe_memes_out_v2"


# -------------------------------------------------------------------
# MODEL LOADING AND IMAGE GENERATION
# -------------------------------------------------------------------

def load_model():
    """
    Load SDXL Turbo once and return the pipeline.
    This model is much lighter than SD3 medium and should fit on a 16 GB T4.
    """

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if DEVICE == "cuda":
        pipe = pipe.to(DEVICE)
        pipe.enable_attention_slicing("max")

    return pipe


def generate_background(pipe, caption: str) -> np.ndarray:
    """
    Use SDXL Turbo to generate a meme background and return it as BGR array.
    """

    prompt = (
        f'A clean simple meme style scene that matches the caption: "{caption}". '
        "Keep the image friendly and neutral. "
        "No offensive stereotypes."
    )

    # SDXL Turbo is designed for very few steps
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            height=512,
            width=512,
        )

    pil_img = result.images[0]          # PIL RGB
    img_rgb = np.array(pil_img)         # H x W x 3 RGB
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


# -------------------------------------------------------------------
# TEXT CLEANING AND OPENCV OVERLAY
# -------------------------------------------------------------------

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


def overlay_caption_opencv(img_bgr: np.ndarray, caption: str) -> np.ndarray:
    """
    Draw a single line caption at the bottom of the image using OpenCV.
    """
    caption = caption.strip()
    if not caption:
        return img_bgr

    h, w, _ = img_bgr.shape

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(1.0, h / 800.0 * 1.2)
    thickness = max(2, int(h / 400.0))

    (text_w, text_h), baseline = cv2.getTextSize(caption, font, scale, thickness)

    x = (w - text_w) // 2
    y = h - 40

    bar_top = max(0, y - text_h - 20)
    cv2.rectangle(img_bgr, (0, bar_top), (w, h), (0, 0, 0), -1)

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

            image_prompt = safe_text or "a harmless humorous situation"

            try:
                bg_bgr = generate_background(pipe, image_prompt)

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

