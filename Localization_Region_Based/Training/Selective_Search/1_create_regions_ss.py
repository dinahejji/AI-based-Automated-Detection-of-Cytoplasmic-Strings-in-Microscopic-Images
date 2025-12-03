# --------------------------------------------
# Selective Search for IVF embryo dataset
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • OUT_DIR/viz/<stem>.png
#   • OUT_DIR/yolo/<stem>.txt        (YOLO: cls cx cy w h angle)
#   • OUT_DIR/coco_annotations.json  (optional)
# --------------------------------------------

import os
import sys
import json
import glob
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from aux_crop_images import icm_roi_from_path
except Exception:
    icm_roi_from_path = None  # optional; set USE_ROI=False if you don't have this

# -------- CONFIG --------
IMAGE_DIR = "PATH TO YOUR IMAGES"

OUT_DIR     = "PATH TO OUTPUT FOLDER"
SAVE_VIZ    = True
EXPORT_YOLO = True             # writes YOLO labels: cls cx cy w h angle(deg)
EXPORT_COCO = False            # COCO AABB export (if you want it)

# thresholds / speed
MIN_RECT_AREA = 100            # px^2; skip tiny proposals early

YOLO_CLASS_ID     = 0
ANGLE_IN_DEGREES  = True       # Selective Search proposals are axis-aligned -> angle = 0.0

# --- Mode: full image OR one ROI ---
USE_ROI  = False                # set True to run only on ROI_XYWH
ROI_XYWH = ()                  # (x,y,w,h) on original image when USE_ROI=True

# --- Selective Search params ---
SS_MODE           = "quality"  # 'fast' or 'quality'
SS_TOP_K          = 2000        # cap number of raw SS rects per image/crop
SS_MERGE_MIN_SIZE = 70        # lower -> more small regions (more proposals)

# -------- Determinism ---------
def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_deterministic(42)

# -------- HELPERS: IO --------
def safe_load_rgb(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# -------- HELPERS: drawing & formatting --------
def obb_to_yolo_line(cx, cy, w, h, ang, W, H, cls_id=0, degrees=True):
    ang_out = ang if degrees else float(np.deg2rad(ang))
    return f"{cls_id} {cx/W:.6f} {cy/H:.6f} {w/W:.6f} {h/H:.6f} {ang_out:.6f}"

def draw_obb(image_rgb: np.ndarray, obb_array: np.ndarray):
    img = image_rgb.copy()
    for cx, cy, w, h, ang in obb_array:
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(ang))
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(img, [box], isClosed=True, color=(0,255,0), thickness=2)
    return img

# -------- COCO STRUCTURE (optional) --------
def init_coco():
    return {"images": [], "annotations": [], "categories": [{"id": 1, "name": "object"}]}

def add_coco_image(coco, image_id, file_name, width, height):
    coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height})

def add_coco_ann(coco, ann_id, image_id, bbox_xywh, area, score, category_id=1):
    coco["annotations"].append({
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [float(b) for b in bbox_xywh],
        "area": float(area),
        "iscrowd": 0,
        "score": float(score),
    })

# -------- Selective Search (plain) --------
def ss_proposals(img_rgb: np.ndarray,
                 mode: str = SS_MODE,
                 top_k: int = SS_TOP_K,
                 merge_min_size: int = SS_MERGE_MIN_SIZE):
    """
    Run OpenCV Selective Search on the given RGB image (no resizing/letterboxing).
    Returns: list of dicts: {"rect": (x,y,w,h), "score": float}
    Score is area/img_area heuristic (OpenCV SS has no confidence).
    """
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
    img_area = float(H * W)
    out = []
    for (x, y, w, h) in rects:
        if w <= 0 or h <= 0:
            continue
        area = w * h
        if area < MIN_RECT_AREA:
            continue
        score = area / img_area  # simple heuristic
        out.append({"rect": (int(x), int(y), int(w), int(h)), "score": float(score)})
    return out

# -------- MAIN --------
def main():
    global ROI_XYWH
    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_VIZ:
        os.makedirs(f"{OUT_DIR}/viz", exist_ok=True)
    if EXPORT_YOLO:
        os.makedirs(f"{OUT_DIR}/yolo", exist_ok=True)

    image_paths = sorted(sum([glob.glob(os.path.join(IMAGE_DIR, p))
                              for p in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")], []))
    if not image_paths:
        raise SystemExit(f"No images found under {IMAGE_DIR}")

    coco = init_coco() if EXPORT_COCO else None
    ann_id = 1

    print("Selective Search mode:", SS_MODE)

    for img_idx, img_path in enumerate(tqdm(image_paths, desc="Processing")):
        if USE_ROI:
            if icm_roi_from_path is None:
                raise RuntimeError("ROI requested but aux_crop_images.icm_roi_from_path not available.")
            ROI_XYWH = icm_roi_from_path(img_path, return_normalized=True)['bbox_xywh']

        img_rgb = safe_load_rgb(img_path)
        if img_rgb is None:
            print(f"[skip] unreadable image: {img_path}")
            stem = Path(img_path).stem
            if EXPORT_YOLO:
                open(f"{OUT_DIR}/yolo/{stem}.txt", "w").close()
            continue

        H, W = img_rgb.shape[:2]
        stem = Path(img_path).stem

        obbs_global = []
        scores = []

        if USE_ROI:
            # --- Single-ROI path (plain crop) ---
            rx, ry, rw, rh = ROI_XYWH
            rx = int(max(0, min(rx, W-1)))
            ry = int(max(0, min(ry, H-1)))
            rw = int(max(1, min(rw, W - rx)))
            rh = int(max(1, min(rh, H - ry)))

            crop = img_rgb[ry:ry+rh, rx:rx+rw]
            proposals = ss_proposals(crop, mode=SS_MODE, top_k=SS_TOP_K, merge_min_size=SS_MERGE_MIN_SIZE)

            # AABB -> OBB (angle=0) -> GLOBAL by offsetting ROI origin
            for p in proposals:
                x, y, w, h = p["rect"]
                cx, cy = (rx + x + w/2.0), (ry + y + h/2.0)
                obbs_global.append([cx, cy, float(w), float(h), 0.0])
                scores.append(p["score"])

        else:
            # --- Full-image path ---
            proposals = ss_proposals(img_rgb, mode=SS_MODE, top_k=SS_TOP_K, merge_min_size=SS_MERGE_MIN_SIZE)
            for p in proposals:
                x, y, w, h = p["rect"]
                cx, cy = x + w/2.0, y + h/2.0
                obbs_global.append([cx, cy, float(w), float(h), 0.0])
                scores.append(p["score"])

        # --- If nothing found, export empties and continue
        if len(obbs_global) == 0:
            if EXPORT_YOLO:
                open(f"{OUT_DIR}/yolo/{stem}.txt", "w").close()
            if EXPORT_COCO:
                add_coco_image(coco, img_idx+1, os.path.basename(img_path), W, H)
            if SAVE_VIZ:
                cv2.imwrite(f"{OUT_DIR}/viz/{stem}.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            continue

        # --- export YOLO-OBB (normalized to ORIGINAL W,H) ---
        if EXPORT_YOLO:
            lines = [
                obb_to_yolo_line(cx, cy, ww, hh, ang, W, H, YOLO_CLASS_ID, degrees=ANGLE_IN_DEGREES)
                for (cx, cy, ww, hh, ang) in obbs_global
            ]
            with open(f"{OUT_DIR}/yolo/{stem}.txt", "w") as f:
                f.write("\n".join(lines))

        # --- (optional) export COCO as AABB ---
        if EXPORT_COCO:
            add_coco_image(coco, img_idx+1, os.path.basename(img_path), W, H)
            for (cx, cy, ww, hh, ang), s in zip(obbs_global, scores):
                rect = ((float(cx), float(cy)), (float(ww), float(hh)), float(ang))
                pts = cv2.boxPoints(rect)
                x1, y1 = pts[:,0].min(), pts[:,1].min()
                x2, y2 = pts[:,0].max(), pts[:,1].max()
                add_coco_ann(coco, ann_id, img_idx+1, [x1, y1, x2-x1, y2-y1], ww*hh, float(s), category_id=1)
                ann_id += 1

        # --- visualization ---
        if SAVE_VIZ:
            vis = draw_obb(img_rgb, np.asarray(obbs_global, dtype=np.float32))
            cv2.imwrite(f"{OUT_DIR}/viz/{stem}.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    if EXPORT_COCO and coco is not None:
        with open(f"{OUT_DIR}/coco_annotations.json", "w") as f:
            json.dump(coco, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
