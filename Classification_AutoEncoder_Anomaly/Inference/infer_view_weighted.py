#!/usr/bin/env python3
"""
Inference + visualization for Weighted AE v2.

- Loads trained AE (weighted v2)
- Runs inference on the test split
- Uses a fixed threshold to classify:
    TP: string correctly detected
    TN: normal correctly rejected
    FP: normal misclassified as string
    FN: string missed
- For a few examples from each category, shows:
    * Original image (resized)
    * Residual heatmap
    * Masked residual (low-texture region)
    * Overlay of masked residual on original

Press any key to go to next image, ESC to exit.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.feature import graycomatrix, graycoprops

# =========================================================
# CONFIG
# =========================================================
DATA_ROOT = "cs_strings_dataset"
IMG_SIZE = 256
CKPT = "ae_weighted_best_v2.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# threshold you used for weighted AE v2 (change if needed)
THRESHOLD = 0.307355

# how many samples to show from each category
N_SHOW_PER_CLASS = 3

# =========================================================
# GLCM CONFIG & HELPERS
# =========================================================
CFG = dict(
    clahe_clip=2.0,
    clahe_grid=8,
    bilateral_d=5,
    bilateral_sigma_color=12,
    bilateral_sigma_space=7,

    glcm_levels=32,
    glcm_win=31,
    glcm_step=8,
    glcm_distances=(1, 2),
    glcm_angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    glcm_prop="contrast",
    glcm_interp_sigma=1.5,

    kmeans_attempts=5,
)


def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    imgs = [p for p in imgs if "overlay" not in p.name.lower()]
    return sorted(imgs)


def preprocess_u8(bgr, out_size):
    """BGR -> grayscale u8 with CLAHE + bilateral, resized to out_size."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if out_size is not None:
        g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(
        clipLimit=CFG["clahe_clip"],
        tileGridSize=(CFG["clahe_grid"], CFG["clahe_grid"]),
    )
    g = clahe.apply(g)
    g = cv2.bilateralFilter(
        g,
        CFG["bilateral_d"],
        CFG["bilateral_sigma_color"],
        CFG["bilateral_sigma_space"],
    )
    return g


def glcm_heat_windowed(gray_u8):
    """Compute windowed GLCM heat (0..1 float) on u8 image (H,W)."""
    H, W = gray_u8.shape
    L = CFG["glcm_levels"]
    q = np.floor(gray_u8.astype(np.float32) / (256.0 / L)).astype(np.uint8)

    win = CFG["glcm_win"]
    step = CFG["glcm_step"]
    if win % 2 == 0:
        win += 1
    half = win // 2

    ys = np.arange(half, H - half, step)
    xs = np.arange(half, W - half, step)
    heat_coarse = np.zeros((len(ys), len(xs)), np.float32)

    for iy, y in enumerate(ys):
        y0, y1 = y - half, y + half + 1
        row = q[y0:y1, :]
        for ix, x in enumerate(xs):
            x0, x1 = x - half, x + half + 1
            patch = row[:, x0:x1]
            glcm = graycomatrix(
                patch,
                distances=CFG["glcm_distances"],
                angles=CFG["glcm_angles"],
                levels=L,
                symmetric=True,
                normed=True,
            )
            prop = graycoprops(glcm, CFG["glcm_prop"]).mean()
            heat_coarse[iy, ix] = prop

    lo, hi = np.percentile(heat_coarse, 1), np.percentile(heat_coarse, 99)
    hc = np.clip((heat_coarse - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
    heat = cv2.resize(hc, (W, H), interpolation=cv2.INTER_CUBIC)
    if CFG["glcm_interp_sigma"] and CFG["glcm_interp_sigma"] > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), CFG["glcm_interp_sigma"])
    return heat


def kmeans_on_heat(heat, k=2):
    """Return (labels_ord, centers_ord, (mask0, mask1)) with centers ascending."""
    H, W = heat.shape
    X = heat.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    _, labels, centers = cv2.kmeans(
        X, k, None, criteria, CFG["kmeans_attempts"], cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(H, W)
    centers = centers.reshape(-1)
    order = np.argsort(centers)
    remap = np.zeros_like(order)
    remap[order] = np.arange(k)
    labels_ord = remap[labels]
    centers_ord = centers[order]
    masks = [(labels_ord == i) for i in range(k)]
    return labels_ord.astype(np.uint8), centers_ord.astype(np.float32), masks


# =========================================================
# DATASET
# =========================================================
class CSStringsDatasetWeighted(Dataset):
    """
    Weighted AE dataset:
        test -> data_root/test/{normal, abnormal}
    Returns:
      x: [1,H,W] float32 in [0,1]
      W: [1,H,W] float32 weight map
      y: 0 (normal) or 1 (abnormal)
      path: image path (str)
    """

    def __init__(self, data_root, split, img_size=256, seed=None):
        super().__init__()
        self.img_size = img_size
        self.split = split
        root = Path(data_root)

        if split == "train":
            self.paths = list_images(root / "train" / "normal")
            self.labels = [0] * len(self.paths)
        elif split in ("val", "test"):
            normal = list_images(root / split / "normal")
            abnormal = list_images(root / split / "abnormal")
            self.paths = normal + abnormal
            self.labels = [0] * len(normal) + [1] * len(abnormal)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self._cache = {}  # path -> (x01, W)

    def __len__(self):
        return len(self.paths)

    def _compute_x_and_W(self, path):
        if path in self._cache:
            return self._cache[path]

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        g = preprocess_u8(bgr, out_size=self.img_size)  # uint8
        heat = glcm_heat_windowed(g)
        _, _, (mask_low, mask_high) = kmeans_on_heat(heat, k=2)

        # Clean low-texture mask with erosion
        mask_low = mask_low.astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask_low = cv2.erode(mask_low, kernel, iterations=1)
        mask_low = mask_low.astype(np.float32)  # 0/1

        # Build soft weights: 1 outside, 1.5 inside, then normalize mean to 1
        W_raw = 1.0 + 0.5 * mask_low  # [H,W] {1.0, 1.5}
        W_raw = W_raw[None, ...].astype(np.float32)  # [1,H,W]
        W_mean = W_raw.mean()
        W = (W_raw / (W_mean + 1e-8)).astype(np.float32)

        x01 = (g.astype(np.float32) / 255.0)[None, ...]

        self._cache[path] = (x01, W)
        return x01, W

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        y = self.labels[idx]
        x01, W = self._compute_x_and_W(path)
        return torch.from_numpy(x01), torch.from_numpy(W), int(y), path


# =========================================================
# MODEL
# =========================================================
class ConvAE(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bottleneck(z)
        xhat = self.dec(z)
        return xhat


# =========================================================
# LOSS HELPERS (for residual maps)
# =========================================================
def ssim_loss(x, y, C1=0.01**2, C2=0.03**2, win=11):
    pad = win // 2
    mu_x = F.avg_pool2d(x, win, 1, pad)
    mu_y = F.avg_pool2d(y, win, 1, pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(x * x, win, 1, pad) - mu_x2
    sigma_y = F.avg_pool2d(y * y, win, 1, pad) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, win, 1, pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2)
    ssim = num / (den + 1e-8)
    return (1.0 - ssim).clamp(0, 2)


def grad_l1(x, y):
    kx = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    gx_x = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy_x = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    gx_y = F.conv2d(y, kx, padding=1, groups=y.shape[1])
    gy_y = F.conv2d(y, ky, padding=1, groups=y.shape[1])

    return (gx_x - gx_y).abs() + (gy_x - gy_y).abs()


def compute_residual_map(x, xhat):
    """Combine L1 + 0.2 SSIM + 0.1 Grad into pixelwise residual map."""
    L1 = torch.sqrt((x - xhat) ** 2 + 1e-6)
    SSIM = ssim_loss(x, xhat)
    GRAD = grad_l1(x, xhat)
    resid = 1.0 * L1 + 0.2 * SSIM + 0.1 * GRAD
    return resid[0, 0].detach().cpu().numpy()  # HxW


# =========================================================
# SCORE EVALUATION (same as training script)
# =========================================================
@torch.no_grad()
def eval_scores_weighted(dloader, model, device):
    """Compute anomaly scores (P95 residual inside low-GLCM region)."""
    model.eval()
    scores, labels, paths = [], [], []

    for x, W, y, path in dloader:
        x = x.to(device, non_blocking=True).float()
        W = W.to(device, non_blocking=True).float()
        xhat = model(x)

        low_mask = (W > 1.0).float()  # [B,1,H,W]

        L1 = torch.sqrt((x - xhat) ** 2 + 1e-6)
        SSIM = ssim_loss(x, xhat)
        GRAD = grad_l1(x, xhat)
        resid = 1.0 * L1 + 0.2 * SSIM + 0.1 * GRAD  # [B,1,H,W]

        B = x.size(0)
        for b in range(B):
            lm = low_mask[b] > 0.5
            r_b = resid[b]
            if lm.any():
                r_vals = r_b[lm]
            else:
                r_vals = r_b.flatten()
            score = torch.quantile(r_vals, 0.95)
            scores.append(float(score.item()))
            labels.append(int(y[b]))
            paths.append(path[b])

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32), paths


# =========================================================
# VISUALIZATION
# =========================================================
def show_heatmap(path, img_bgr, resid, mask_low, title_extra=""):
    """
    Shows:
        - original (resized to AE input size)
        - residual heatmap
        - masked residual heatmap
        - overlay heatmap on image
    """
    H, W = resid.shape

    # resize original BGR image to match residual size
    img_bgr_resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    # normalize residual to [0,255]
    r = resid.copy()
    r = r - r.min()
    r = r / (r.max() + 1e-8)
    r255 = (r * 255).astype(np.uint8)

    # ensure mask has same size
    if mask_low.shape != r255.shape:
        mask_low = cv2.resize(
            mask_low.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
    mask = mask_low.astype(np.uint8)

    masked = r255 * mask

    heat = cv2.applyColorMap(r255, cv2.COLORMAP_JET)
    heat_masked = cv2.applyColorMap(masked, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr_resized, 0.6, heat_masked, 0.7, 0)

    window_prefix = f"{title_extra} " if title_extra else ""
    cv2.imshow(window_prefix + "Original", img_bgr_resized)
    cv2.imshow(window_prefix + "Residual Heatmap", heat)
    cv2.imshow(window_prefix + "Masked Residual", heat_masked)
    cv2.imshow(window_prefix + "Overlay", overlay)

    print(f"\n[{title_extra}] {path}")
    print("Press any key for next image, ESC to quit.")

    key = cv2.waitKey(0)
    if key == 27:  # ESC
        cv2.destroyAllWindows()
        exit()
    cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================
def main():
    # load model
    device = torch.device(DEVICE)
    model = ConvAE(in_ch=1).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[info] Loaded model from {CKPT}")

    # dataset + loader
    test_ds = CSStringsDatasetWeighted(DATA_ROOT, "test", img_size=IMG_SIZE)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)
    print(f"[info] Test samples: {len(test_ds)}")

    # run inference to get scores
    scores, labels, paths = eval_scores_weighted(test_loader, model, device)
    preds = (scores >= THRESHOLD).astype(np.int32)

    # collect indices for TP/TN/FP/FN
    TP_idx, TN_idx, FP_idx, FN_idx = [], [], [], []
    for i, (y, p) in enumerate(zip(labels, preds)):
        if y == 1 and p == 1:
            TP_idx.append(i)
        elif y == 0 and p == 0:
            TN_idx.append(i)
        elif y == 0 and p == 1:
            FP_idx.append(i)
        elif y == 1 and p == 0:
            FN_idx.append(i)

    print(f"TP={len(TP_idx)}, TN={len(TN_idx)}, FP={len(FP_idx)}, FN={len(FN_idx)}")
    rng = np.random.default_rng(1337)

    # helper to visualize a few samples from a list of indices
    def show_from_indices(indices, label):
        if not indices:
            print(f"[warn] No samples for {label}")
            return
        chosen = rng.choice(indices, size=min(N_SHOW_PER_CLASS, len(indices)), replace=False)
        for idx in chosen:
            path = paths[idx]
            gt = labels[idx]
            scr = scores[idx]
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue

            # recompute x, W for this sample
            x, W, _, _ = test_ds[idx]
            x = x.unsqueeze(0).to(device).float()
            W_t = W.to(device).float()

            with torch.no_grad():
                xhat = model(x)

            resid = compute_residual_map(x, xhat)
            mask_low = (W_t[0, 0].cpu().numpy() > 1.0).astype(np.uint8)

            title = f"{label} (score={scr:.3f}, y={gt})"
            show_heatmap(path, img_bgr, resid, mask_low, title_extra=title)

    # show few examples from each category
    show_from_indices(TP_idx, "TP (string correctly detected)")
    show_from_indices(TN_idx, "TN (normal correctly rejected)")
    show_from_indices(FP_idx, "FP (false alarm)")
    show_from_indices(FN_idx, "FN (missed string)")

    print("[done]")


if __name__ == "__main__":
    main()
