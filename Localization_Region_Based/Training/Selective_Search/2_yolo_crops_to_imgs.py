# --------------------------------------------
# Crops images into region proposals 
# Author: Mauro Mendez Mora
# --------------------------------------------
# Outputs:
#   • OUT_FOLDER/EMBRYO/region_#.png
# --------------------------------------------

import os
import cv2
import numpy as np
from pathlib import Path

# --------------- CONFIG ---------------
IMAGE_FOLDER = "PATH TO ORIGINAL IMAGES"
LABEL_FOLDER  = "PATH TO YOLO FOLDER CREATED ON 1_create_regions_ss.py"  # YOLO-OBB txt with: cls cx cy w h angle
OUT_FOLDER = "PATH TO OUTPUT FOLDER"
ANGLE_IN_RADIANS = False     # set True if your angle column is in radians
ANGLE_IS_CLOCKWISE = True          # Set True if your angles are clockwise (will negate)
BORDER_VALUE = (0, 0, 0)            # fill for out-of-image areas
# ---------------------------------------

def read_yolo_obb(path):
    """Yield (cls, cx_n, cy_n, w_n, h_n, angle_deg_raw) from YOLO-OBB txt."""
    with open(path, "r") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                print(f"[warn] line {ln}: expected 6 fields, got {len(parts)} -> skip")
                continue
            try:
                cls_id = int(float(parts[0]))
                cx, cy, w, h, ang = map(float, parts[1:6])
                ang_deg = np.degrees(ang) if ANGLE_IN_RADIANS else ang
                if ANGLE_IS_CLOCKWISE:
                    ang_deg = -ang_deg
                yield cls_id, cx, cy, w, h, float(ang_deg)
            except Exception as e:
                print(f"[warn] line {ln}: parse error: {e} -> skip")

def ordered_corners_from_rotated_rect(cx, cy, w, h, angle_deg):
    """
    Build TL, TR, BR, BL corners from center-size-angle without touching aspect ratio.
    (cx,cy,w,h) are in pixels; angle is degrees, CCW-positive.
    """
    hw, hh = w * 0.5, h * 0.5
    # local corners (x right, y down): TL, TR, BR, BL
    local = np.array([[-hw, -hh],
                      [ hw, -hh],
                      [ hw,  hh],
                      [-hw,  hh]], dtype=np.float32)
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]], dtype=np.float32)
    world = (local @ R.T) + np.array([cx, cy], dtype=np.float32)
    return world.astype(np.float32)  # TL, TR, BR, BL

def rectify_crop(image, corners_tl_tr_br_bl, w, h):
    """
    Perspective-warp the quadrilateral to a (w x h) axis-aligned patch.
    """
    dst_w = max(int(round(w)), 1)
    dst_h = max(int(round(h)), 1)
    dst = np.array([[0, 0],
                    [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1],
                    [0, dst_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_tl_tr_br_bl, dst)
    patch = cv2.warpPerspective(
        image, M, (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BORDER_VALUE
    )
    return patch

def main():
    for IMAGE_PATH in Path(IMAGE_FOLDER).glob("*"):
        if not IMAGE_PATH.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            continue
        img_name = IMAGE_PATH.stem
        LABEL_PATH = Path(LABEL_FOLDER) / f"{img_name}.txt"
        if not LABEL_PATH.exists():
            print(f"[warn] label file not found for image {IMAGE_PATH.name} -> skip")
            continue

    
        img = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise SystemExit(f"Could not read image: {IMAGE_PATH}")
        H, W = img.shape[:2]

        out_dir = Path(OUT_FOLDER) / Path(IMAGE_PATH.parent.name) / Path(Path(IMAGE_PATH).stem).with_name(f"{img_name}_ICM_crop")
        os.makedirs(out_dir, exist_ok=True)

        count = 0
        for cls_id, cx_n, cy_n, w_n, h_n, ang_deg in read_yolo_obb(LABEL_PATH):
            # denormalize to pixels (no swaps, preserve aspect exactly as labeled)
            cx = cx_n * W
            cy = cy_n * H
            ww = max(w_n * W, 1.0)
            hh = max(h_n * H, 1.0)

            # build corners and rectify
            corners = ordered_corners_from_rotated_rect(cx, cy, ww, hh, ang_deg)
            patch = rectify_crop(img, corners, ww, hh)

            out_name = f"{img_name}_ICM_crop_{count:03d}.png"
            out_path = str(out_dir / out_name)
            ok = cv2.imwrite(out_path, patch)
            if not ok:
                print(f"[warn] failed to write {out_path}")
            count += 1

        print(f"Saved {count} crops to {out_dir}")

if __name__ == "__main__":
    main()