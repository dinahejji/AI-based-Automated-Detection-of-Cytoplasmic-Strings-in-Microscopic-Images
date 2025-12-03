#!/usr/bin/env python3
"""
The unweighted Convolutional AE baseline for CS detection.

- Same architecture as the weighted AE
- the difference from weighted version:
  * No GLCM, no low-texture mask, no spatial weighting
  * Loss is a simple mean over all pixels

- Early stopping:
  * Choose best epoch by AUROC on val (normal vs abnormal) based on
    reconstruction error.

- Threshold:
  * Youden's J on validation ROC
  * Use same threshold on test set.
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

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    imgs = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    imgs = [p for p in imgs if "overlay" not in p.name.lower()]
    return sorted(imgs)

# -------------------------- Dataset -------------------------- #
class CSStringsDatasetUnweighted(Dataset):
    def __init__(self, data_root, split, img_size=256):
        super().__init__()
        self.img_size = img_size
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx])
        label = self.labels[idx]

        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)

        g = preprocess_u8(bgr, out_size=self.img_size)
        x01 = (g.astype(np.float32) / 255.0)[None, ...]  # [1,H,W]

        return torch.from_numpy(x01), int(label), path

# -------------------------- Model (same as weighted) -------------------------- #
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
        # Bottleneck (pure conv, no FC)
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

# -------------------------- Losses (SSIM, GradL1, Charbonnier) -------------------------- #
def ssim_loss(x, y, C1=0.01**2, C2=0.03**2, win=11):
    """
    Per-pixel (1 - SSIM) using mean and variance over a sliding window.
    This implementation returns a map with same spatial size as x.
    """
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
    """Per-pixel L1 difference between Sobel gradients of x and y."""
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

def recon_combo_loss_unweighted(x, xhat, alpha=1.0, beta=0.2, gamma=0.1):
    """
    Same combination as in weighted AE, but no spatial weighting.
    Returns:
      batch_mean_loss, per_sample_scores
    """
    L1 = charbonnier(x - xhat)          # [B,1,H,W]
    SSIM = ssim_loss(x, xhat)           # [B,1,H,W]
    GRAD = grad_l1(x, xhat)             # [B,1,H,W]

    # Per-sample mean over spatial dims (and channels)
    per_sample = alpha * L1.mean(dim=(1, 2, 3)) \
               + beta  * SSIM.mean(dim=(1, 2, 3)) \
               + gamma * GRAD.mean(dim=(1, 2, 3))

    return per_sample.mean(), per_sample

# -------------------------- Metrics & evaluation -------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_scores(dloader, model, device):
    """
    Compute reconstruction-based scores (per image) and labels.
    Score = mean over pixels of combined residual (Charbonnier + 0.2*(1-SSIM) + 0.1*GradL1).
    """
    model.eval()
    scores = []
    labels = []
    paths = []

    for x, y, path in dloader:
        x = x.to(device, non_blocking=True).float()
        xhat = model(x)
        _, per_sample = recon_combo_loss_unweighted(x, xhat)
        scores.extend(per_sample.cpu().numpy().tolist())
        labels.extend([int(v) for v in y])
        paths.extend(path)

    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32), paths

def pick_threshold_from_val(scores, labels):
    """
    Threshold = argmax (TPR - FPR) (Youden's J) on ROC curve.
    """
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
    ap.add_argument("--ckpt-out", default="ae_unweighted_best.pt")
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Datasets
    train_ds = CSStringsDatasetUnweighted(args.data_root, "train", img_size=args.img_size)
    val_ds   = CSStringsDatasetUnweighted(args.data_root, "val",   img_size=args.img_size)
    test_ds  = CSStringsDatasetUnweighted(args.data_root, "test",  img_size=args.img_size)

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

        for x, _, _ in train_loader:
            x = x.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            xhat = model(x)
            loss, _ = recon_combo_loss_unweighted(x, xhat)
            loss.backward()
            opt.step()
            running.append(loss.item())

        # Validation scores
        val_scores, val_labels, _ = eval_scores(val_loader, model, device)
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
    val_scores, val_labels, _ = eval_scores(val_loader, model, device)
    thr, (fpr_j, tpr_j) = pick_threshold_from_val(val_scores, val_labels)
    val_metrics = compute_metrics(val_scores, val_labels, thr)

    print("\n=== Validation (unweighted AE) ===")
    print(f"Chosen threshold (Youden): {thr:.6f}  (TPR={tpr_j:.3f}, FPR={fpr_j:.3f})")
    print(f"AUROC: {val_metrics['AUROC']:.4f}")
    print(f"Acc: {val_metrics['Acc']:.4f}  Prec: {val_metrics['Prec']:.4f}  Rec: {val_metrics['Rec']:.4f}  F1: {val_metrics['F1']:.4f}")
    print(f"Confusion  TP:{val_metrics['TP']}  FP:{val_metrics['FP']}  TN:{val_metrics['TN']}  FN:{val_metrics['FN']}")

    # ---------------------- Test evaluation ---------------------- #
    test_scores, test_labels, _ = eval_scores(test_loader, model, device)
    test_metrics = compute_metrics(test_scores, test_labels, thr)

    print("\n=== Test (unweighted AE) ===")
    print(f"Threshold used: {thr:.6f}")
    print(f"AUROC: {test_metrics['AUROC']:.4f}")
    print(f"Acc: {test_metrics['Acc']:.4f}  Prec: {test_metrics['Prec']:.4f}  Rec: {test_metrics['Rec']:.4f}  F1: {test_metrics['F1']:.4f}")
    print(f"Confusion  TP:{test_metrics['TP']}  FP:{test_metrics['FP']}  TN:{test_metrics['TN']}  FN:{test_metrics['FN']}")

if __name__ == "__main__":
    main()
