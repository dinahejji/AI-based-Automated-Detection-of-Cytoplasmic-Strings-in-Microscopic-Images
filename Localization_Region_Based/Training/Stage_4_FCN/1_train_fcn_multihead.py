"""
Two-head MLP (regularized) with epoch-wise balanced sampling, MixUp,
and early stopping + best checkpoint based on validation AP.

Head 1: classification (Focal BCE)
Head 2: IoU regression (Smooth L1 to df['IoU'])

Each epoch:
  - Build a NEW balanced train dataframe by undersampling the majority class
    from IMBALANCED_TRAIN_CSV, then train on that for the epoch.

Validation AP drives model selection & early stopping.
Also logs: total loss, per-head losses, accuracy, IoU MAE, AP (train/val).

Author: Mauro Mendez Mora
--------------------------------------------
  Outputs:
    • CSV with classification labels
--------------------------------------------
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score  # for AP

# =========================
# GLOBALS (edit these)
# =========================
FEATURES_DIR = "PATH TO FEATURES FOLDER (OUTPUT OF STAGE 2/STEP 1).csv"
OUT_DIR = "PATH TO OUTPUT FOLDER"

# Validation set
val_df = pd.read_csv('PATH TO BALANCED CLASSIFICATION LABELS CSV FILE (OUTPUT STAGE 3/STEP 3) FOR VAL SET.csv')

# Imbalanced pool (used to create a fresh balanced train set each epoch)
IMBALANCED_TRAIN_CSV = 'PATH TO CLASSIFICATION LABELS CSV FILE (OUTPUT STAGE 3/STEP 2) FOR TRAIN SET.csv'
imbalanced_pool_df = pd.read_csv(IMBALANCED_TRAIN_CSV)

# Training hyperparams
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
PATIENCE = 10              # early stop on val AP (no improvement for this many epochs)
SEED = 42
ALPHA = 0.25               # focal alpha
GAMMA = 2.0                # focal gamma
FEATURE_DIM = 768
DROPOUT_P = 0.3            # dropout probability in trunk

# Loss weights
LAMBDA_CLS = 1.0
LAMBDA_IOU = 1.0

# Epoch balancing config
EPOCH_BALANCE_MAX_PER_CLASS = None  # e.g., 2000 to cap each class per epoch; None = auto(min count)

# MixUp
MIXUP_ALPHA = 0.2          # 0 = off; typical 0.1–0.4
MIXUP_PROB = 0.5           # chance to apply MixUp per batch (0–1)

# =========================
# UTILITIES
# =========================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_npz_path(features_dir: Path, img_name: str, b1_idx: int, set_name: str) -> Path:
    """
    Your custom path layout:
    {FEATURES_DIR}/{set_name}/{IMG_Name_base}_ICM_crop/{IMG_Name_base}_ICM_crop_{b1_idx:03d}.npz
    """
    base = img_name[:-5]
    p = features_dir / set_name / f"{base}_ICM_crop" / f"{base}_ICM_crop_{b1_idx:03d}.npz"
    if p.exists():
        return p
    raise FileNotFoundError(f"NPZ not found: {p}")


def load_dino_npz(path: str):
    """
    Load DINOv3 global embedding from .npz (key: 'global_emb').
    DINOv3 global features are already L2-normalized — don't re-standardize.
    """
    data = np.load(path, allow_pickle=False)
    if "global_emb" in data:
        return data["global_emb"]
    # Fallback if stored differently
    return data.get("global_emb")


class FeaturesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features_dir: Path, feature_dim: int = 768, set_name="train"):
        self.df = df.reset_index(drop=True)
        self.features_dir = Path(features_dir)
        self.feature_dim = feature_dim
        self.set_name = set_name

        required = {"IMG_Name", "b1_idx", "Class_GT", "IoU"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["IMG_Name"])
        b1_idx = int(str(row["b1_idx"]))
        y_cls = float(row["Class_GT"])
        y_iou = float(row["IoU"])

        npz_path = resolve_npz_path(self.features_dir, img_name, b1_idx, self.set_name)
        feat = load_dino_npz(npz_path)

        if feat.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Feature length mismatch in {npz_path} (len={feat.shape[-1]}), expected {self.feature_dim}"
            )

        x = torch.from_numpy(feat)                          # (768,)
        y_cls = torch.tensor([y_cls], dtype=torch.float32)  # (1,)
        y_iou = torch.tensor([y_iou], dtype=torch.float32)  # (1,)
        return x, y_cls, y_iou


class TwoHeadMLP(nn.Module):
    """
    Shared trunk with LayerNorm + Dropout -> two heads:
      - cls_head: binary classification logit
      - iou_head: IoU regression scalar
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


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # logits/targets: (N,1)
        bce_loss = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = alpha_t * focal_factor * bce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def make_epoch_balanced_df(pool_df: pd.DataFrame, epoch: int, max_per_class=None) -> pd.DataFrame:
    """
    Create a NEW balanced dataframe by undersampling the majority class.
    Re-seeded per epoch for fresh samples.
    """
    rng = np.random.RandomState(SEED + epoch)
    pos_df = pool_df[pool_df["Class_GT"] == 1]
    neg_df = pool_df[pool_df["Class_GT"] == 0]
    n_pos, n_neg = len(pos_df), len(neg_df)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot balance: one class has zero samples in the pool_df.")
    per_class = min(n_pos, n_neg)
    if max_per_class is not None:
        per_class = min(per_class, int(max_per_class))
    pos_sample = pos_df.sample(n=per_class, replace=(per_class > n_pos), random_state=rng)
    neg_sample = neg_df.sample(n=per_class, replace=(per_class > n_neg), random_state=rng)
    epoch_df = pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return epoch_df


def safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AP (AUC-PR) safely. Returns np.nan if only one class present.
    y_true: (N,) in {0,1}; y_score: (N,) probabilities
    """
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def mixup(x, y_cls, y_iou, alpha=MIXUP_ALPHA):
    """Feature-space MixUp. Works with BCE (soft labels) and SmoothL1."""
    if alpha is None or alpha <= 0.0:
        return x, y_cls, y_iou
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_m = lam * x + (1 - lam) * x[idx]
    y_cls_m = lam * y_cls + (1 - lam) * y_cls[idx]   # soft target for BCE
    y_iou_m = lam * y_iou + (1 - lam) * y_iou[idx]
    return x_m, y_cls_m, y_iou_m


@torch.no_grad()
def evaluate(model, loader, device, focal_loss, smoothl1):
    """
    Returns:
      total, cls_loss, iou_loss, acc, iou_mae, ap
    """
    model.eval()
    tot_loss = tot_cls = tot_iou = 0.0
    tot_acc = 0
    tot_mae = 0.0
    n_examples = 0

    all_probs = []
    all_targets = []

    for xb, yb_cls, yb_iou in loader:
        xb = xb.to(device, non_blocking=True)
        yb_cls = yb_cls.to(device, non_blocking=True)
        yb_iou = yb_iou.to(device, non_blocking=True)

        logits, iou_pred = model(xb)
        cls_loss = focal_loss(logits, yb_cls)
        iou_loss = smoothl1(iou_pred, yb_iou)
        loss = LAMBDA_CLS * cls_loss + LAMBDA_IOU * iou_loss

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        acc = (preds == yb_cls).sum().item()
        mae = torch.abs(iou_pred - yb_iou).mean().item()

        # Collect for AP
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(yb_cls.detach().cpu().numpy())

        bs = xb.size(0)
        tot_loss += loss.item() * bs
        tot_cls  += cls_loss.item() * bs
        tot_iou  += iou_loss.item() * bs
        tot_acc  += acc
        tot_mae  += mae * bs
        n_examples += bs

    y_score = np.concatenate(all_probs, axis=0).ravel()
    y_true  = np.concatenate(all_targets, axis=0).ravel()
    ap = safe_average_precision(y_true, y_score)

    return (
        tot_loss / n_examples,
        tot_cls  / n_examples,
        tot_iou  / n_examples,
        tot_acc  / n_examples,
        tot_mae  / n_examples,
        ap,
    )


def train_with_epoch_balancing(pool_df: pd.DataFrame, val_df: pd.DataFrame):
    set_seed(SEED)

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Dataloaders
    val_ds = FeaturesDataset(val_df, FEATURES_DIR, feature_dim=FEATURE_DIM, set_name="val")
    num_workers = max(2, min(8, os.cpu_count() or 2))
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadMLP(in_dim=FEATURE_DIM, p_drop=DROPOUT_P).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    focal = FocalBCEWithLogitsLoss(alpha=ALPHA, gamma=GAMMA)
    smoothl1 = nn.SmoothL1Loss(reduction="mean")

    history = {
        "train_total_loss": [], "val_total_loss": [],
        "train_cls_loss": [],   "val_cls_loss": [],
        "train_iou_loss": [],   "val_iou_loss": [],
        "train_acc": [],        "val_acc": [],
        "train_iou_mae": [],    "val_iou_mae": [],
        "train_ap": [],         "val_ap": [],
        "train_size": [], "train_pos": [], "train_neg": [],
    }

    # Early stopping on VAL AP (maximize)
    best_val_ap = -float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0

    print("Training")
    for epoch in range(1, EPOCHS + 1):
        # --------- New balanced DataFrame/DataLoader this epoch ---------
        epoch_df = make_epoch_balanced_df(
            pool_df=pool_df,
            epoch=epoch,
            max_per_class=EPOCH_BALANCE_MAX_PER_CLASS,
        )
        n_pos = int((epoch_df["Class_GT"] == 1).sum())
        n_neg = int((epoch_df["Class_GT"] == 0).sum())

        train_ds = FeaturesDataset(epoch_df, FEATURES_DIR, feature_dim=FEATURE_DIM, set_name="train")
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
        )

        # --------- Train ---------
        model.train()
        run_total = run_cls = run_iou = 0.0
        run_acc = 0
        run_mae = 0.0
        n_seen = 0
        all_train_probs = []
        all_train_targets = []

        for xb, yb_cls, yb_iou in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb_cls = yb_cls.to(device, non_blocking=True)
            yb_iou = yb_iou.to(device, non_blocking=True)

            # MixUp with probability
            if MIXUP_ALPHA > 0 and np.random.rand() < MIXUP_PROB and xb.size(0) > 1:
                xb, yb_cls, yb_iou = mixup(xb, yb_cls, yb_iou, alpha=MIXUP_ALPHA)

            optimizer.zero_grad(set_to_none=True)
            logits, iou_pred = model(xb)

            cls_loss = focal(logits, yb_cls)
            iou_loss = smoothl1(iou_pred, yb_iou)
            loss = LAMBDA_CLS * cls_loss + LAMBDA_IOU * iou_loss

            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            acc = (preds == yb_cls).sum().item()
            mae = torch.abs(iou_pred - yb_iou).mean().item()

            all_train_probs.append(probs.detach().cpu().numpy())
            all_train_targets.append(yb_cls.detach().cpu().numpy())

            bs = xb.size(0)
            run_total += loss.item() * bs
            run_cls   += cls_loss.item() * bs
            run_iou   += iou_loss.item() * bs
            run_acc   += acc
            run_mae   += mae * bs
            n_seen    += bs

        # Train epoch averages
        train_total = run_total / n_seen
        train_cls   = run_cls   / n_seen
        train_iou   = run_iou   / n_seen
        train_acc   = run_acc   / n_seen
        train_mae   = run_mae   / n_seen
        ytr_score = np.concatenate(all_train_probs, axis=0).ravel()
        ytr_true  = np.concatenate(all_train_targets, axis=0).ravel()
        train_ap = safe_average_precision(ytr_true, ytr_score)

        # --------- Validation (drives early stopping) ---------
        val_total, val_cls, val_iou, val_acc, val_mae, val_ap = evaluate(model, val_loader, device, focal, smoothl1)

        # Log history
        history["train_total_loss"].append(train_total)
        history["val_total_loss"].append(val_total)
        history["train_cls_loss"].append(train_cls)
        history["val_cls_loss"].append(val_cls)
        history["train_iou_loss"].append(train_iou)
        history["val_iou_loss"].append(val_iou)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_iou_mae"].append(train_mae)
        history["val_iou_mae"].append(val_mae)
        history["train_ap"].append(train_ap)
        history["val_ap"].append(val_ap)
        history["train_size"].append(int(len(epoch_df)))
        history["train_pos"].append(n_pos)
        history["train_neg"].append(n_neg)

        train_ap_str = f"{train_ap:.4f}" if not np.isnan(train_ap) else "nan"
        val_ap_str   = f"{val_ap:.4f}" if not np.isnan(val_ap) else "nan"

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Tot {train_total:.4f} | Val Tot {val_total:.4f} || "
            f"Train (cls {train_cls:.4f}, iou {train_iou:.4f}, acc {train_acc:.4f}, iouMAE {train_mae:.4f}, AP {train_ap_str}) | "
            f"Val   (cls {val_cls:.4f}, iou {val_iou:.4f}, acc {val_acc:.4f}, iouMAE {val_mae:.4f}, AP {val_ap_str})"
        )

        # ----- Early stopping on VAL AP (maximize) -----
        improved = (val_ap > best_val_ap + 1e-8) if not np.isnan(val_ap) else False
        if improved:
            best_val_ap = val_ap
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_ap": val_ap,
                "val_total_loss": val_total,
                "val_cls_loss": val_cls,
                "val_iou_loss": val_iou,
                "val_acc": val_acc,
                "val_iou_mae": val_mae,
                "config": {
                    "lr": LR, "weight_decay": WEIGHT_DECAY,
                    "alpha": ALPHA, "gamma": GAMMA,
                    "lambda_cls": LAMBDA_CLS, "lambda_iou": LAMBDA_IOU,
                    "batch_size": BATCH_SIZE, "seed": SEED,
                    "dropout_p": DROPOUT_P,
                    "mixup_alpha": MIXUP_ALPHA, "mixup_prob": MIXUP_PROB,
                },
            }
            epochs_no_improve = 0
            # save immediately
            best_path = Path(OUT_DIR) / "best_model.pt"
            torch.save(best_state, best_path)
            print(f"  ✓ Saved best checkpoint at epoch {epoch} with val AP={val_ap_str} -> {best_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val AP improvement for {PATIENCE} epochs). "
                  f"Best epoch: {best_state['epoch'] if best_state else 'N/A'}, "
                  f"best val AP: {best_val_ap:.4f}")
            break

    # ---------- Plots ----------
    out_dir = Path(OUT_DIR)

    # Total loss
    plt.figure()
    plt.plot(range(1, len(history["train_total_loss"]) + 1), history["train_total_loss"], label="train")
    plt.plot(range(1, len(history["val_total_loss"]) + 1), history["val_total_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss"); plt.title("Total Loss over epochs"); plt.legend()
    plt.savefig(out_dir / "total_loss.png", bbox_inches="tight"); plt.close()

    # Classification accuracy
    plt.figure()
    plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"], label="train")
    plt.plot(range(1, len(history["val_acc"]) + 1), history["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Classification Accuracy over epochs"); plt.legend()
    plt.savefig(out_dir / "cls_accuracy.png", bbox_inches="tight"); plt.close()

    # IoU MAE
    plt.figure()
    plt.plot(range(1, len(history["train_iou_mae"]) + 1), history["train_iou_mae"], label="train")
    plt.plot(range(1, len(history["val_iou_mae"]) + 1), history["val_iou_mae"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title("IoU MAE over epochs"); plt.legend()
    plt.savefig(out_dir / "iou_mae.png", bbox_inches="tight"); plt.close()

    # AP over epochs
    plt.figure()
    plt.plot(range(1, len(history["train_ap"]) + 1), history["train_ap"], label="train")
    plt.plot(range(1, len(history["val_ap"]) + 1), history["val_ap"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Average Precision (AP)")
    plt.title("AP over epochs"); plt.legend()
    plt.savefig(out_dir / "ap.png", bbox_inches="tight"); plt.close()

    # Save history (includes AP, per-class sizes)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_history.csv", index=False)
    print(f"Saved plots and history to: {out_dir}")


if __name__ == "__main__":
    train_with_epoch_balancing(imbalanced_pool_df, val_df)
