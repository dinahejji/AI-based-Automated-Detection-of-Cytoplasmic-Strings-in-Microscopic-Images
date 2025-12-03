"""
Single-image inference:
 - Selective Search proposals on the input image
 - DINOv3 global embeddings for each proposal (batched)
 - Two-head MLP (cls + IoU): score = sigmoid(cls_logit) * pred_iou
 - Choose best region, draw on the original image, and save visualization

Author: Mauro Mendez Mora
"""

import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel

# =========================
# CONFIG
# =========================

# Input/output
INPUT_IMAGE = "PATH TO IMAGE FILE"              # image to process
INPUT_LABELS = "PATH TO IMAGE'S GROUND TRUTH LABELS FILE (YOLO STYLE TXT FILE)"  # for visualization
OUT_DIR     = "PATH TO OUTPUT FOLDER/"
OUT_NAME    = INPUT_IMAGE.split('/')[-1]                 # file saved under OUT_DIR

# Trained checkpoint 
CHECKPOINT  = "PATH TO TRAINED TWO-HEAD MODEL CHECKPOINT FILE.pth (OUTPUT OF Training/STAGE 4/STEP 1)"

# DINOv3 model
DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# Model / feature details (must match training)
FEATURE_DIM = 768
DROPOUT_P   = 0.3
USE_IOU_SIGMOID = False     # set True if you sigmoid'ed IoU during training
CLAMP_IOU       = True      # clamp IoU prediction to [0, 1] if head is unbounded

# Selective Search params
SS_MODE           = "quality"   # 'fast' or 'quality'
SS_TOP_K          = 2000        # cap number of proposals (set smaller/larger as needed)
SS_MERGE_MIN_SIZE = 70          # smaller => more small regions
MIN_RECT_AREA     = 100         # skip tiny regions (px^2)

# DINO batching
EMB_BATCH_SIZE = 64             # proposals per forward pass (tune based on VRAM)
NUM_WORKERS    = 0              # not used; everything in-process
SEED           = 42

# =========================
# UTILITIES
# =========================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def safe_load_rgb(path: str) -> np.ndarray:
    """Robust image loader returning RGB uint8 (H,W,3) or raises."""
    # Handle paths with unicode / spaces
    arr = np.fromfile(path, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def ss_proposals(img_rgb: np.ndarray,
                 mode: str = SS_MODE,
                 top_k: int = SS_TOP_K,
                 merge_min_size: int = SS_MERGE_MIN_SIZE) -> List[Tuple[int, int, int, int]]:
    """
    Selective Search on RGB image; returns list of (x, y, w, h) proposals.
    """
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError("OpenCV contrib module missing. Install: pip install opencv-contrib-python")

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    if mode == "quality":
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
    try:
        ss.setMergeMinSize(int(merge_min_size))
    except Exception:
        pass

    rects = ss.process()  # list of (x,y,w,h)
    if top_k is not None and top_k > 0:
        rects = rects[:top_k]

    H, W = img_rgb.shape[:2]
    out = []
    for (x, y, w, h) in rects:
        if w <= 0 or h <= 0:
            continue
        if w * h < MIN_RECT_AREA:
            continue
        # keep inside image bounds
        x = int(max(0, min(x, W - 1)))
        y = int(max(0, min(y, H - 1)))
        w = int(max(1, min(w, W - x)))
        h = int(max(1, min(h, H - y)))
        out.append((x, y, w, h))
    return out

def crop_to_pil(img_rgb: np.ndarray, rects: List[Tuple[int,int,int,int]]) -> List[Image.Image]:
    """
    Crop (x,y,w,h) from np RGB image to PIL images (robust RGB).
    """
    crops = []
    H, W = img_rgb.shape[:2]
    for (x, y, w, h) in rects:
        x2, y2 = x + w, y + h
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x2), min(H, y2)
        crop = img_rgb[y1:y2, x1:x2]
        pil = Image.fromarray(crop)
        if pil.mode != "RGB":
            pil = ImageOps.grayscale(pil).convert("RGB")
        crops.append(pil)
    return crops

def draw_rect(ax, xyxy, color, label=None, lw=3):
    x1, y1, x2, y2 = xyxy
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=lw))
    if label is not None:
        ax.text(x1, max(0, y1 - 5), label, fontsize=10, color=color,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

# =========================
# MODELS
# =========================

class TwoHeadMLP(nn.Module):
    """
    Shared trunk with LayerNorm + Dropout -> cls + iou heads.
    Must match the architecture you trained.
    """
    def __init__(self, in_dim=768, p_drop=0.3):
        super().__init__()
        self.norm_in = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.cls_head = nn.Linear(128, 1)
        self.iou_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.norm_in(x)
        h = self.trunk(x)
        return self.cls_head(h), self.iou_head(h)

# =========================
# INFERENCE
# =========================

def run_inference_single_image():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load image & proposals
    rgb = safe_load_rgb(INPUT_IMAGE)
    H, W = rgb.shape[:2]
    rects = ss_proposals(rgb, mode=SS_MODE, top_k=SS_TOP_K, merge_min_size=SS_MERGE_MIN_SIZE)
    if len(rects) == 0:
        raise SystemExit("No proposals from Selective Search.")

    # 2) Prepare DINOv3
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    dinov3 = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(device)

    # 3) Build crops & embed with DINOv3 (batched)
    crops = crop_to_pil(rgb, rects)
    all_embs = []  # (N, FEATURE_DIM)
    with torch.inference_mode():
        for i in range(0, len(crops), EMB_BATCH_SIZE):
            batch_imgs = crops[i:i+EMB_BATCH_SIZE]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            print(batch_imgs[0].size)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = dinov3(**inputs)
            print(outputs.pooler_output.size())
            
            # Global embedding
            global_emb = outputs.pooler_output  # [B, D]
            all_embs.append(global_emb.detach().cpu())
    feats = torch.cat(all_embs, dim=0)  # (N, D)
    print(feats.size())
    
    if feats.shape[1] != FEATURE_DIM:
        raise ValueError(f"DINO feature dim {feats.shape[1]} != expected {FEATURE_DIM}")

    # 4) Load your two-head model + checkpoint
    model = TwoHeadMLP(in_dim=FEATURE_DIM, p_drop=DROPOUT_P).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # 5) Score proposals: sigmoid(logit) * pred_iou
    with torch.inference_mode():
        x = feats.to(device)
        logits, iou_pred = model(x)              # (N,1), (N,1)
        prob = torch.sigmoid(logits).squeeze(1)  # (N,)
        if USE_IOU_SIGMOID:
            iou_hat = torch.sigmoid(iou_pred).squeeze(1)
        else:
            iou_hat = iou_pred.squeeze(1)
            if CLAMP_IOU:
                iou_hat = torch.clamp(iou_hat, 0.0, 1.0)
        score = prob * iou_hat                   # (N,)

    # 6) Pick best region & visualize
    best_idx = int(torch.argmax(score).item())
    x, y, w, h = rects[best_idx]
    x1, y1, x2, y2 = x, y, x + w, y + h

    annotations = []
    with open(INPUT_LABELS, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            class_id = int(parts[0])
            x_c, y_c, w_, h_ = map(float, parts[1:])
            annotations.append((class_id, x_c, y_c, w_, h_))
    print(INPUT_LABELS, annotations)



    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.axis("off")
    # label = f"best score={score[best_idx].item():.3f}\np={prob[best_idx].item():.3f}, iou={iou_hat[best_idx].item():.3f}"
    draw_rect(ax, (x1, y1, x2, y2), color="blue", label="Pred", lw=3)
    
    
    for ann in annotations:
        class_id, x_c, y_c, w, h = ann
        bw = w * W
        bh = h * H
        cx = x_c * W
        cy = y_c * H
        print(cx, cy, bw, bh)

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        draw_rect(ax, (x1, y1, x2, y2), color="lime", label="GT", lw=3)
        print((x1, y1, x2, y2))
    
    
    out_path = Path(OUT_DIR) / OUT_NAME
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Optional: also save a simple text summary
    with open(Path(OUT_DIR) / "prediction_summary.txt", "w") as f:
        f.write(f"Image: {INPUT_IMAGE}\n")
        f.write(f"Image size: {W}x{H}\n")
        f.write(f"Proposals: {len(rects)}\n")
        f.write(f"Best box (x1,y1,x2,y2): {x1},{y1},{x2},{y2}\n")
        f.write(f"Best prob: {prob[best_idx].item():.6f}\n")
        f.write(f"Best iou: {iou_hat[best_idx].item():.6f}\n")
        f.write(f"Best score: {score[best_idx].item():.6f}\n")

    print(f"Saved visualization to: {out_path}")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_inference_single_image()
