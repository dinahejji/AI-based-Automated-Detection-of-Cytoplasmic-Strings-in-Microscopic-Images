# --------------------------------------------
# Calculates IoU between aligned bounding boxes and selective search proposals 
# Regular IoU between:
#   b0: YOLO AABB  <cls> cx cy w h
#   b1: YOLO AABB  <cls> cx cy w h [ignored
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • CSV with IoU results
# --------------------------------------------

import glob, os
from dataclasses import dataclass
from typing import List
import pandas as pd
from PIL import Image

# ================= CONFIG =================
IMAGES_DIR = f"PATH TO ORIGINAL IMAGES FOLDER/"
B0_DIR     = f"PATH TO LABELS FOLDER (BOUNDING BOXES)"  # b0: <cls> cx cy w h
B1_DIR     = f"PATH TO YOLO FOLDER WITH PROPOSAL BOUNDING BOXES (OUTPUT OF STAGE 1)"  # b1: <cls> cx cy w h <ignored>
OUT_CSV    = f"NAME OF OUTPUT CSV FILE.csv"
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
# ==========================================

@dataclass
class AABB:
    xmin: float; ymin: float; xmax: float; ymax: float

def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else (1.0 if v > 1 else v)

def yolo_line_to_aabb(line: str, W: int, H: int) -> AABB:
    """
    Parse a YOLO AABB line with at least 4 numeric values after class:
      <cls> cx cy w h [extras...]
    Returns pixel-space AABB.
    """
    parts = line.strip().split()
    if len(parts) < 1 + 4:
        raise ValueError(f"Malformed YOLO line: {line}")
    cx_n, cy_n, w_n, h_n = map(float, parts[1:1+4])
    cx = _clamp01(cx_n) * W
    cy = _clamp01(cy_n) * H
    w  = max(0.0, min(1.0, w_n)) * W
    h  = max(0.0, min(1.0, h_n)) * H
    hw, hh = w/2.0, h/2.0
    return AABB(cx - hw, cy - hh, cx + hw, cy + hh)

def iou(a: AABB, b: AABB) -> float:
    ix = max(0.0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
    iy = max(0.0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
    inter = ix * iy
    if inter <= 0: return 0.0
    area_a = max(0.0, a.xmax - a.xmin) * max(0.0, a.ymax - a.ymin)
    area_b = max(0.0, b.xmax - b.xmin) * max(0.0, b.ymax - b.ymin)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def load_yolo_file(path: str, W: int, H: int) -> List[AABB]:
    boxes: List[AABB] = []
    if not os.path.exists(path): return boxes
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"): continue
            try:
                boxes.append(yolo_line_to_aabb(line, W, H))
            except Exception as e:
                print(f"[WARN] skip {path}:{ln_no} -> {e}")
    return boxes

def infer_pair_path(base_dir: str, stem: str) -> str:
    for ext in [".txt", ".yolo"]:
        p = os.path.join(base_dir, f"{stem}{ext}")
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(base_dir, f"{stem}.*"))
    return hits[0] if hits else os.path.join(base_dir, f"{stem}.txt")

def main():
    rows = []
    for root, _, files in os.walk(IMAGES_DIR):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMAGE_EXTS:
                continue
            img_path = os.path.join(root, fn)
            stem = os.path.splitext(fn)[0]
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception as e:
                print(f"[IMG WARN] Skipping {img_path}: {e}")
                continue

            b0_path = infer_pair_path(B0_DIR, stem)
            b1_path = infer_pair_path(B1_DIR, stem)

            b0_boxes = load_yolo_file(b0_path, W, H)   # b0: cx cy w h
            b1_boxes = load_yolo_file(b1_path, W, H)   # b1: cx cy w h [ignored extra]

            for i, gt in enumerate(b0_boxes):
                for j, pr in enumerate(b1_boxes):
                    rows.append({
                        "IMG_Name": fn,
                        "b0_idx": f"{i:03d}",
                        "b1_idx": f"{j:03d}",
                        "IoU": iou(gt, pr),
                        # --- added coordinates for later use ---
                        "b0_xmin": gt.xmin, "b0_ymin": gt.ymin,
                        "b0_xmax": gt.xmax, "b0_ymax": gt.ymax,
                        "b1_xmin": pr.xmin, "b1_ymin": pr.ymin,
                        "b1_xmax": pr.xmax, "b1_ymax": pr.ymax,
                    })

    df = pd.DataFrame(rows, columns=[
        "IMG_Name", "b0_idx", "b1_idx", "IoU",
        "b0_xmin", "b0_ymin", "b0_xmax", "b0_ymax",
        "b1_xmin", "b1_ymin", "b1_xmax", "b1_ymax",
    ])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()
