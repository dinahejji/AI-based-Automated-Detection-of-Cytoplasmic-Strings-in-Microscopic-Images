#!/usr/bin/env python3
"""
Improved Convolutional Autoencoder with GLCM-based weighting for CS detection.

Changes vs. previous weighted version:
  - GLCM low-texture mask is cleaned with erosion.
  - Weighting is softer: W_raw = 1 + 0.5 * low_mask, then normalized so mean(W) ≈ 1.
  - Training loss uses W (spatially weighted Charbonnier + SSIM + GradL1).
  - Evaluation score does NOT use W directly:
        - Residual is computed without W.
        - Score = P95 of residual inside the low-GLCM region (fallback to global P95 if mask empty).
  - Threshold still chosen by Youden's J on validation ROC, then applied to test.

"""

import argparse, random, time
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, precision_recall_fscore_support, roc_curve
)

# -------------------------- GLCM Config -------------------------- #
CFG = dict(
    # Preprocessing
    clahe_clip=2.0, clahe_grid=8,
    bilateral_d=5, bilateral_sigma_color=12, bilateral_sigma_space=7,

    # GLCM heat
    glcm_levels=32,
    glcm_win=31,
    glcm_step=8,
    glcm_distances=(1, 2),
    glcm_angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
    glcm_prop="contrast",
    glcm_interp_sigma=1.5,

    # KMeans
    kmeans_attempts=5,
)

# -------------------------- Preprocessing -------------------------- #
def preprocess_u8(bgr, out_size):
    """BGR -> grayscale u8 with CLAHE + bilateral, resized to out_size."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if out_size is not None:
        g = cv2.resize(g, (out_size, out_size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=CFG["clahe_clip"],
                            tileGridSize=(CFG["clahe_grid"], CFG["clahe_grid"]))
    g = clahe.apply(g)
    g = cv2.bilateralFilter(g, CFG["bilateral_d"],
                            CFG["bilateral_sigma_color"],
                            CFG["bilateral_sigma_space"])
    return g

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    imgs = [p for p in imgs if "overlay" not in p.name.lower()]
    return sorted(imgs)

# -------------------------- GLCM & mask helpers -------------------------- #
from skimage.feature import graycomatrix, graycoprops

def glcm_heat_windowed(gray_u8):
    """Compute windowed GLCM heat (0..1 float) on u8 image (H,W)."""
    H, W = gray_u8.shape
    L = CFG["glcm_levels"]
    q = np.floor(gray_u8.astype(np.float32) / (256.0 / L)).astype(np.uint8)

    win = CFG["glcm_win"]; step = CFG["glcm_step"]
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
                symmetric=True, normed=True
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

# -------------------------- Dataset (weighted) -------------------------- #
class CSStringsDatasetWeighted(Dataset):
    """
    Weighted AE dataset:

    split = 'train' -> data_root/train/normal
    split = 'val'   -> data_root/val/{normal, abnormal}
    split = 'test'  -> data_root/test/{normal, abnormal}

    Returns:
      x: [1,H,W] float32 in [0,1] (preprocessed grayscale)
      W: [1,H,W] float32 weight map (soft GLCM-based)
      y: 0 (normal) or 1 (abnormal)
      path: image path
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
        W_raw = 1.0 + 0.5 * mask_low     # [H,W], values in {1.0, 1.5}
        W_raw = W_raw[None, ...].astype(np.float32)  # [1,H,W]
        W_mean = W_raw.mean()
        W = (W_raw / (W_mean + 1e-8)).astype(np.float32)  # normalize

        x01 = (g.astype(np.float32) / 255.0)[None, ...]  # [1,H,W]

        self._cache[path] = (x01, W)
        return x01, W

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        y = self.labels[idx]
        x01, W = self._compute_x_and_W(path)
        return torch.from_numpy(x01), torch.from_numpy(W), int(y), path

# -------------------------- Model (same as before) -------------------------- #
class ConvAE(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1), nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),

            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.enc(x)
        z = self.bottleneck(z)
        xhat = self.dec(z)
        return xhat

# -------------------------- Losses -------------------------- #
def ssim_loss(x, y, C1=0.01**2, C2=0.03**2, win=11):
    """Per-pixel (1 - SSIM) using mean/variance over sliding window."""
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
    """Per-pixel L1 difference between Sobel gradients."""
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    gx_x = F.conv2d(x, kx, padding=1, groups=x.shape[1])
    gy_x = F.conv2d(x, ky, padding=1, groups=x.shape[1])
    gx_y = F.conv2d(y, kx, padding=1, groups=y.shape[1])
    gy_y = F.conv2d(y, ky, padding=1, groups=y.shape[1])

    return (gx_x - gx_y).abs() + (gy_x - gy_y).abs()

def charbonnier(residual, eps=1e-3):
    return torch.sqrt(residual * residual + eps * eps)

def weighted_mean(t, W):
    """Spatial mean with weights W (broadcast over channels if needed)."""
    if W.dim() == 4 and W.size(1) == 1 and t.size(1) > 1:
        W = W.repeat(1, t.size(1), 1, 1)
    return (t * W).sum(dim=(1, 2, 3)) / (W.sum(dim=(1, 2, 3)) + 1e-8)

def recon_combo_loss_weighted(x, xhat, W, alpha=1.0, beta=0.2, gamma=0.1):
    L1   = charbonnier(x - xhat)
    SSIM = ssim_loss(x, xhat)
    GRAD = grad_l1(x, xhat)
    per_sample = (alpha * weighted_mean(L1, W) +
                  beta  * weighted_mean(SSIM, W) +
                  gamma * weighted_mean(GRAD, W))
    return per_sample.mean(), per_sample  # (batch_mean, per-sample scores)

# -------------------------- Metrics & eval -------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_scores_weighted(dloader, model, device):
    """
    Evaluation-time scores for weighted AE:
      - residual combo computed WITHOUT W,
      - score = P95 of residual inside low-GLCM region (deduced from W),
        fallback to global P95 if mask empty.
    """
    model.eval()
    scores, labels, paths = [], [], []

    for x, W, y, path in dloader:
        x = x.to(device, non_blocking=True).float()
        W = W.to(device, non_blocking=True).float()
        xhat = model(x)

        # Recover low-mask from W: W > 1 => low-texture region
        low_mask = (W > 1.0).float()  # [B,1,H,W]

        # Compute residual components (no weighting)
        L1   = charbonnier(x - xhat)
        SSIM = ssim_loss(x, xhat)
        GRAD = grad_l1(x, xhat)
        resid = 1.0*L1 + 0.2*SSIM + 0.1*GRAD  # [B,1,H,W]

        B = x.size(0)
        for b in range(B):
            lm = low_mask[b] > 0.5  # bool
            r_b = resid[b]          # [1,H,W]

            if lm.any():
                # use only values inside low-mask
                r_vals = r_b[lm]
            else:
                # fallback to all pixels
                r_vals = r_b.flatten()

            score = torch.quantile(r_vals, 0.95)
            scores.append(float(score.item()))
            labels.append(int(y[b]))
            paths.append(path[b])

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32), paths

def pick_threshold_from_val(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    k = np.argmax(j)
    return thr[k], (fpr[k], tpr[k])

def compute_metrics(scores, labels, thr):
    y_pred = (scores >= thr).astype(np.int32)

    if len(np.unique(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = float('nan')

    tn, fp, fn, tp = confusion_matrix(labels, y_pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, y_pred, average='binary', zero_division=0
    )
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    return dict(
        AUROC=auroc, Acc=acc, Prec=prec, Rec=rec, F1=f1,
        TP=tp, FP=fp, TN=tn, FN=fn, thr=thr
    )

# -------------------------- Main -------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="cs_strings_dataset")
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ckpt-out", default="ae_weighted_best_v2.pt")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Datasets
    train_ds = CSStringsDatasetWeighted(args.data_root, "train", img_size=args.img_size, seed=args.seed)
    val_ds   = CSStringsDatasetWeighted(args.data_root, "val",   img_size=args.img_size, seed=args.seed+1)
    test_ds  = CSStringsDatasetWeighted(args.data_root, "test",  img_size=args.img_size, seed=args.seed+2)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )

    print(f"[info] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"[info] Device: {device}")

    # Model + optimizer
    model = ConvAE(in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auroc = -1.0
    best_state = None
    best_epoch = -1

    # ---------------------- Training loop ---------------------- #
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        t0 = time.time()

        for x, W, _, _ in train_loader:
            x = x.to(device, non_blocking=True).float()
            W = W.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            xhat = model(x)
            loss, _ = recon_combo_loss_weighted(x, xhat, W)
            loss.backward()
            opt.step()
            running.append(loss.item())

        # Validation with weighted scoring logic
        val_scores, val_labels, _ = eval_scores_weighted(val_loader, model, device)
        if len(np.unique(val_labels)) > 1:
            val_auroc = roc_auc_score(val_labels, val_scores)
        else:
            val_auroc = float('nan')

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | train_loss={np.mean(running):.4f} | val_AUROC={val_auroc:.4f} | {dt:.1f}s")

        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch

    # ---------------------- Load best & save ---------------------- #
    if best_state is not None:
        torch.save(best_state, args.ckpt_out)
        print(f"[best] Saved best model (epoch {best_epoch}, val AUROC={best_val_auroc:.4f}) -> {args.ckpt_out}")
        model.load_state_dict(best_state, strict=True)
    else:
        print("[warn] No improvement recorded; using last model.")

    # ---------------------- Threshold from validation ---------------------- #
    val_scores, val_labels, _ = eval_scores_weighted(val_loader, model, device)
    thr, (fpr_j, tpr_j) = pick_threshold_from_val(val_scores, val_labels)
    val_metrics = compute_metrics(val_scores, val_labels, thr)

    print("\n=== Validation (weighted AE v2) ===")
    print(f"Chosen threshold (Youden): {thr:.6f}  (TPR={tpr_j:.3f}, FPR={fpr_j:.3f})")
    print(f"AUROC: {val_metrics['AUROC']:.4f}")
    print(f"Acc: {val_metrics['Acc']:.4f}  Prec: {val_metrics['Prec']:.4f}  Rec: {val_metrics['Rec']:.4f}  F1: {val_metrics['F1']:.4f}")
    print(f"Confusion  TP:{val_metrics['TP']}  FP:{val_metrics['FP']}  TN:{val_metrics['TN']}  FN:{val_metrics['FN']}")

    # ---------------------- Test evaluation ---------------------- #
    test_scores, test_labels, _ = eval_scores_weighted(test_loader, model, device)
    test_metrics = compute_metrics(test_scores, test_labels, thr)

    print("\n=== Test (weighted AE v2) ===")
    print(f"Threshold used: {thr:.6f}")
    print(f"AUROC: {test_metrics['AUROC']:.4f}")
    print(f"Acc: {test_metrics['Acc']:.4f}  Prec: {test_metrics['Prec']:.4f}  Rec: {test_metrics['Rec']:.4f}  F1: {test_metrics['F1']:.4f}")
    print(f"Confusion  TP:{test_metrics['TP']}  FP:{test_metrics['FP']}  TN:{test_metrics['TN']}  FN:{test_metrics['FN']}")

if __name__ == "__main__":
    main()
