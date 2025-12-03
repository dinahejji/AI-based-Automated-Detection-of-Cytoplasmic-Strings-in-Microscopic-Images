# --------------------------------------------
# Extract DINOv3 features from region proposals 
# Saves as .npz files with global and dense embeddings
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • OUT_FOLDER/EMBRYO/region_#.npz
# --------------------------------------------

import argparse
import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import AutoImageProcessor, AutoModel
from glob import glob
from pathlib import Path
from tqdm import tqdm

# Default: a solid mid-size ViT; swap for larger/smaller variants if needed:
#  - "facebook/dinov3-vitb16-pretrain-lvd1689m"
#  - "facebook/dinov3-vitl16-pretrain-lvd1689m"
#  - "facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"
IMAGE_FOLDER = "PATH TO YOLO CROPPED IMAGES FOLDER/"
OUT_FOLDER = "PATH TO OUTPUT FEATURES FOLDER/"
Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)

def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    # Medical images are often grayscale or paletted; convert robustly to RGB
    if img.mode != "RGB":
        img = ImageOps.grayscale(img).convert("RGB")
    return img

@torch.inference_mode()
def extract_features(image_path: str,
                     processor: AutoImageProcessor,
                     model: AutoModel,
                     device: str,
                     save_npz: str = None):
    img = load_image_rgb(image_path)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # Global embedding (pooled CLS / global token)
    global_emb = outputs.pooler_output[0].cpu()  # [D]

    # Patch embeddings (ViT gives [1, seq_len, hidden])
    last = outputs.last_hidden_state  # [1, seq_len, hidden]
    patch = getattr(model.config, "patch_size", 16)
    h_in = processor.size["height"]
    w_in = processor.size["width"]
    H = h_in // patch
    W = w_in // patch
    dense_emb = last[0, : H * W, :].reshape(H, W, -1).cpu()  # [H, W, D]

    if save_npz:
        Path(save_npz).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_npz,
            global_emb=global_emb.numpy(),      # [D]
            dense_emb=dense_emb.numpy(),        # [H, W, D]
        )

def build_output_path(sub_img: str) -> str:
    sub_img = sub_img.replace('\\', '/')
    out_complete_path = OUT_FOLDER + sub_img.replace(IMAGE_FOLDER, "")
    out_complete_path = "/".join(out_complete_path.split('/')[:-1])  # folder only
    Path(out_complete_path).mkdir(parents=True, exist_ok=True)
    fname = Path(sub_img).stem + ".npz"
    return str(Path(out_complete_path) / fname)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)

    # Collect all sub-images two levels deep (folder → files)
    original_images = sorted(glob(IMAGE_FOLDER + "*"))
    total_files = 0
    per_folder = {}

    for og in original_images:
        subs = sorted(glob(og + "/*"))
        if subs:
            per_folder[og] = subs
            total_files += len(subs)

    # Outer bar for folders, inner bar for images
    with tqdm(total=total_files, desc="Total images", unit="img") as pbar_total:
        for og_img in tqdm(per_folder.keys(), desc="Folders", unit="dir", leave=False):
            sub_images = per_folder[og_img]
            for sub_img in tqdm(sub_images, desc="Processing", unit="img", leave=False):
                out_path = build_output_path(sub_img)
                try:
                    extract_features(sub_img.replace('\\', '/'),
                                     processor, model, device,
                                     save_npz=out_path)
                except Exception as e:
                    # Show the error but keep the bars happy
                    tqdm.write(f"[ERROR] {sub_img}: {e}")
                finally:
                    pbar_total.update(1)
