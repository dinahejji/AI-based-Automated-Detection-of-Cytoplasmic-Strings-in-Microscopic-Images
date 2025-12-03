import os
import io
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModel
import tempfile

# =========================
# CONFIG
# =========================

CHECKPOINT = "PATH TO TRAINED TWO-HEAD MODEL CHECKPOINT FILE.pth (OUTPUT OF Training/STAGE 4/STEP 1)"
DINO_MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

FEATURE_DIM = 768
DROPOUT_P = 0.3
USE_IOU_SIGMOID = False
CLAMP_IOU = True

EMB_BATCH_SIZE = 64
SEED = 42

DISPLAY_SIZE = (500, 500)   # all images shown as 500x500

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

def ss_proposals(
    img_rgb: np.ndarray,
    mode: str = "quality",
    top_k: int = 1200,
    merge_min_size: int = 70,
    min_rect_area: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """Selective Search on RGB image; returns list of (x, y, w, h) proposals."""
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "segmentation"):
        raise RuntimeError(
            "OpenCV contrib module missing. Install: pip install opencv-contrib-python"
        )

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
        if w * h < min_rect_area:
            continue
        # keep inside image bounds
        x = int(max(0, min(x, W - 1)))
        y = int(max(0, min(y, H - 1)))
        w = int(max(1, min(w, W - x)))
        h = int(max(1, min(h, H - y)))
        out.append((x, y, w, h))
    return out


def crop_to_pil(img_rgb: np.ndarray, rects: List[Tuple[int, int, int, int]]) -> List[Image.Image]:
    """Crop (x,y,w,h) from np RGB image to PIL images."""
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
    ax.add_patch(
        plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=lw,
        )
    )
    if label is not None:
        ax.text(
            x1,
            max(0, y1 - 5),
            label,
            fontsize=10,
            color=color,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1),
        )


def fig_to_resized_png_bytes(fig, size=DISPLAY_SIZE):
    """Convert a Matplotlib figure to PNG bytes and resize to given size."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    out_buf = io.BytesIO()
    img.save(out_buf, format="PNG")
    out_buf.seek(0)
    plt.close(fig)
    return out_buf


def resize_np_to_display(img_np: np.ndarray, size=DISPLAY_SIZE) -> np.ndarray:
    """Resize a numpy image (H,W,3) to DISPLAY_SIZE."""
    pil = Image.fromarray(img_np)
    pil = pil.resize(size, Image.BILINEAR)
    return np.array(pil)


def parse_yolo_gt_line(line: str, img_w: int, img_h: int):
    """
    Parse YOLO-style line: cls cx cy w h (normalized to [0,1]).
    Returns (x, y, w, h) in pixel coordinates.
    """
    clean = line.replace("(", " ").replace(")", " ").strip()
    tokens = clean.split()
    if len(tokens) < 5:
        raise ValueError("Expected at least 5 values: cls cx cy w h")

    try:
        _cls = int(float(tokens[0]))
        cx, cy, bw, bh = map(float, tokens[1:5])
    except Exception:
        raise ValueError("Could not parse YOLO line; format must be 'cls cx cy w h'.")

    x = (cx - bw / 2.0) * img_w
    y = (cy - bh / 2.0) * img_h
    w = bw * img_w
    h = bh * img_h

    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    return int(x), int(y), int(w), int(h)


def embedding_heatmap_fig(emb_vec: np.ndarray):
    """
    Create a heatmap figure from a single 768-dim embedding vector.
    Reshape to 24x32 and plot like the example: Global Embedding (768 dims).
    """
    if emb_vec.shape[0] != 768:
        raise ValueError(f"Expected embedding of dim 768, got {emb_vec.shape[0]}")
    grid = emb_vec.reshape(24, 32)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis", interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


# =========================
# MODELS
# =========================

class TwoHeadMLP(nn.Module):
    """Shared trunk with LayerNorm + Dropout -> cls + iou heads."""
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


@st.cache_resource
def load_dino_and_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    dinov3 = AutoModel.from_pretrained(DINO_MODEL_NAME).eval().to(device)
    return processor, dinov3, device


@st.cache_resource
def load_twohead_mlp():
    _, _, device = load_dino_and_device()
    model = TwoHeadMLP(in_dim=FEATURE_DIM, p_drop=DROPOUT_P).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model


# =========================
# INFERENCE LOGIC
# =========================

def run_inference_on_image(
    # pil_image: Image.Image,
    image_path: str,
    ss_mode: str,
    ss_top_k: int,
    ss_merge_min_size: int,
    ss_min_rect_area: int,
):
    set_seed(SEED)
    processor, dinov3, device = load_dino_and_device()
    mlp = load_twohead_mlp()

    # img_rgb = np.array(pil_image.convert("RGB"))
    img_rgb = safe_load_rgb(image_path)

    rects = ss_proposals(
        img_rgb,
        mode=ss_mode,
        top_k=ss_top_k,
        merge_min_size=ss_merge_min_size,
        min_rect_area=ss_min_rect_area,
    )
    if len(rects) == 0:
        raise RuntimeError("No proposals from Selective Search.")

    crops = crop_to_pil(img_rgb, rects)
    all_embs = []

    with torch.inference_mode():
        for i in range(0, len(crops), EMB_BATCH_SIZE):
            batch_imgs = crops[i : i + EMB_BATCH_SIZE]
            inputs = processor(images=batch_imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = dinov3(**inputs)
            global_emb = outputs.pooler_output  # [B, D]
            all_embs.append(global_emb.detach().cpu())

    feats_tensor = torch.cat(all_embs, dim=0)  # (N, D)
    if feats_tensor.shape[1] != FEATURE_DIM:
        raise ValueError(f"DINO feature dim {feats_tensor.shape[1]} != expected {FEATURE_DIM}")

    feats_np = feats_tensor.numpy()

    with torch.inference_mode():
        x = feats_tensor.to(device)
        logits, iou_pred = mlp(x)
        prob = torch.sigmoid(logits).squeeze(1)
        if USE_IOU_SIGMOID:
            iou_hat = torch.sigmoid(iou_pred).squeeze(1)
        else:
            iou_hat = iou_pred.squeeze(1)
            if CLAMP_IOU:
                iou_hat = torch.clamp(iou_hat, 0.0, 1.0)
        score = (prob**0.6) * (iou_hat**0.4)

    best_idx = int(torch.argmax(score).item())

    return {
        "img_rgb": img_rgb,
        "rects": rects,
        "prob": prob.cpu().numpy(),
        "iou_hat": iou_hat.cpu().numpy(),
        "score": score.cpu().numpy(),
        "best_idx": best_idx,
        "feats": feats_np,
    }


# =========================
# STREAMLIT APP
# =========================

st.set_page_config(page_title="CS Detection", layout="wide")

st.title("AI-based Automated Detection of Cytoplasmic Strings in TLM Images​")

st.markdown(
    """
Upload an image, adjust Selective Search parameters, and visualize:

- Selective Search proposals
- DINOv3 global embedding heatmap (768 dims)
- Best proposed region  
- Top-k region proposals

"""
)

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state["results"] = None
    st.session_state["img_size"] = None

uploaded_file = st.file_uploader("TLM image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # pil_img = Image.open(uploaded_file).convert("RGB")
    # img_w, img_h = pil_img.size
    
    # Save uploaded file to a temporary path
    suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name
    
    # Load with the safe loader for size + preview
    img_rgb_preview = safe_load_rgb(image_path)
    img_h, img_w = img_rgb_preview.shape[:2]
    pil_img = Image.fromarray(img_rgb_preview)  # for display only

    # =========================
    # Selective Search parameters
    # =========================
    st.subheader("Selective Search Parameters")

    col_ss1, col_ss2, col_ss3, col_ss4 = st.columns(4)

    with col_ss1:
        ss_mode = st.selectbox(
            "Mode",
            options=["quality", "fast"],
            index=0,
            help="quality = more accurate, slower; fast = faster, fewer regions",
        )

    with col_ss2:
        ss_top_k = st.slider(
            "Max proposals (SS_TOP_K)",
            min_value=100,
            max_value=3000,
            value=1200,
            step=100,
        )

    with col_ss3:
        ss_merge_min_size = st.slider(
            "Merge Min Size",
            min_value=10,
            max_value=300,
            value=70,
            step=10,
            help="Smaller ⇒ more small regions",
        )

    with col_ss4:
        ss_min_rect_area = st.slider(
            "Min Rect Area",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Skip tiny regions (px²)",
        )

    # =========================
    # Ground truth input (YOLO format)
    # =========================
    st.subheader("Ground Truth (YOLO format, optional)")
    use_gt = st.checkbox("Provide ground-truth in YOLO format (cls cx cy w h, normalized)")

    gt_text = ""
    if use_gt:
        gt_text = st.text_area(
            "YOLO label line",
            value="0 0.634540 0.411920 0.038840 0.030840",
            help="Format: class cx cy w h, all in [0,1]. Example: 0 0.5 0.5 0.2 0.2",
        )

    # =========================
    # Run inference (only when button clicked)
    # =========================
    if st.button("Run Inference"):
        try:
            results = run_inference_on_image(
                # pil_image=pil_img,
                image_path=image_path,
                ss_mode=ss_mode,
                ss_top_k=ss_top_k,
                ss_merge_min_size=ss_merge_min_size,
                ss_min_rect_area=ss_min_rect_area,
            )
            st.session_state["results"] = results
            st.session_state["img_size"] = (img_w, img_h)
        except Exception as e:
            st.error(f"Inference failed: {e}")

    # =========================
    # Show outputs if we have results
    # =========================
    if st.session_state["results"] is not None:
        results = st.session_state["results"]
        stored_w, stored_h = st.session_state["img_size"]

        if (img_w, img_h) != (stored_w, stored_h):
            st.warning("Image changed since last inference. Please click **Run Inference** again.")
        else:
            img_rgb = results["img_rgb"]
            rects = results["rects"]
            prob = results["prob"]
            iou_hat = results["iou_hat"]
            score = results["score"]
            best_idx = results["best_idx"]
            feats_np = results["feats"]

            # Parse GT every time (for live overlay update)
            gt_box_pixels = None
            if use_gt and gt_text.strip():
                try:
                    gt_box_pixels = parse_yolo_gt_line(gt_text, img_w, img_h)
                except ValueError as e:
                    st.error(f"Could not parse GT: {e}")
                    gt_box_pixels = None

            # =========================
            # Row: Original / SS boxes / Embedding heatmap
            # =========================
            st.subheader("Overview")

            col1, col2, col3 = st.columns(3)

            # ---- Original (with or without GT) ----
            with col1:
                st.markdown("**Original (with GT if given)**")
                img_orig = img_rgb.copy()
                if gt_box_pixels is not None:
                    gx, gy, gw, gh = gt_box_pixels
                    gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
                    img_bgr = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(img_bgr, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                    img_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(
                    resize_np_to_display(img_orig),
                     width="content",
                )

            # ---- All SS proposals ----
            with col2:
                st.markdown("**All Selective Search Proposals**")
                fig_ss, ax_ss = plt.subplots(figsize=(6, 6))
                ax_ss.imshow(img_rgb)
                ax_ss.axis("off")
                for (x, y, w, h) in rects:
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    draw_rect(ax_ss, (x1, y1, x2, y2), color="tab:blue", lw=0.5)

                if gt_box_pixels is not None:
                    gx, gy, gw, gh = gt_box_pixels
                    gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
                    draw_rect(
                        ax_ss,
                        (gx1, gy1, gx2, gy2),
                        color="tab:green",
                        label="GT",
                        lw=2,
                    )

                fig_ss.tight_layout()
                st.image(
                    fig_to_resized_png_bytes(fig_ss),
                     width="content",
                )

            # ---- Embedding heatmap for best proposal ----
            with col3:
                st.markdown("**DINOv3 Global Embedding (best proposal)**")
                emb_vec = feats_np[best_idx]  # (768,)
                fig_emb = embedding_heatmap_fig(emb_vec)
                st.image(
                    fig_to_resized_png_bytes(fig_emb),
                     width="content",
                )

            # =========================
            # Best proposal overlay
            # =========================
            st.subheader("Best Proposed Region")

            best_x, best_y, best_w, best_h = rects[best_idx]
            best_x1, best_y1 = best_x, best_y
            best_x2, best_y2 = best_x + best_w, best_y + best_h

            fig1, ax1 = plt.subplots(figsize=(6, 6))
            ax1.imshow(img_rgb)
            ax1.axis("off")

            label = (
                f"best score={score[best_idx]:.3f}\n"
                f"p={prob[best_idx]:.3f}, iou={iou_hat[best_idx]:.3f}"
            )
            draw_rect(
                ax1,
                (best_x1, best_y1, best_x2, best_y2),
                color="tab:orange",
                label=label,
                lw=3,
            )

            if gt_box_pixels is not None:
                gx, gy, gw, gh = gt_box_pixels
                gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
                draw_rect(
                    ax1,
                    (gx1, gy1, gx2, gy2),
                    color="tab:green",
                    label="GT",
                    lw=2,
                )

            st.image(
                fig_to_resized_png_bytes(fig1),
                caption="Best proposal (orange) and GT (green, if provided)",
            )

            # =========================
            # Top-k proposals table + overlay
            # =========================
            st.subheader("Top-k Region Proposals (by score)")
            max_k = min(100, len(rects))
            k = st.slider("k (for table & overlay)", 1, max_k, min(10, max_k))

            idx_sorted = np.argsort(-score)
            topk_idx = idx_sorted[:k]

            rows = []
            for idx in topk_idx:
                x, y, w, h = rects[idx]
                rows.append(
                    {
                        "idx": int(idx),
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "prob": float(prob[idx]),
                        "iou_pred": float(iou_hat[idx]),
                        "score": float(score[idx]),
                    }
                )
            df = pd.DataFrame(rows)
            st.dataframe(df,  width="stretch")

            st.subheader("Overlay of Top-k Region Proposals")

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(img_rgb)
            ax2.axis("off")

            for idx in topk_idx:
                x, y, w, h = rects[idx]
                x1, y1, x2, y2 = x, y, x + w, y + h
                draw_rect(ax2, (x1, y1, x2, y2), color="tab:blue", lw=1)

            draw_rect(
                ax2,
                (best_x1, best_y1, best_x2, best_y2),
                color="tab:orange",
                label="Best",
                lw=3,
            )

            if gt_box_pixels is not None:
                gx, gy, gw, gh = gt_box_pixels
                gx1, gy1, gx2, gy2 = gx, gy, gx + gw, gy + gh
                draw_rect(
                    ax2,
                    (gx1, gy1, gx2, gy2),
                    color="tab:green",
                    label="GT",
                    lw=2,
                )

            st.image(
                fig_to_resized_png_bytes(fig2),
                caption="Top-k proposals (blue), best (orange), GT (green)",
            )

else:
    st.info("Please upload an image to begin.")
