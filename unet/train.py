#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet/train.py  (was train_unet.py)

Trains the U-Net (model.py) to map noisy SimAFM images to clean ones, then
scores it against the noisy input and a Gaussian filter (the filter the LAFM
pipeline currently uses) on a held-out test set (70/15/15 train/val/test).

Run: F5 in Spyder, or `python unet/train.py` from any folder.
Edit the CONFIG block below to change the split, model size etc.

This script trains and evaluates only; it makes NO figures. All plots are
made afterwards by analysis/make_figures.py from the files saved here, so
you can restyle figures without retraining.

Outputs (in results/<run_name>/):
- best_model.pt    : weights + the settings needed to reload them
- history.csv      : one row per validation check (losses, RMSE in nm,
                     learning rate, best so far); used for the loss curve
- split.csv        : which rows went to train / val / test
- metrics.csv      : TEST-set RMSE (nm) per tip radius / cone angle / noise
- config.json      : the CONFIG used, plus the scale and data folder

Target choice: this is option 1 from the plan, the target is the noise-free
image with the SAME tip (clean_path in data_table.csv). For option 2
(target = noise-free, smaller tip) the only change is which clean file each
row is paired with.

@author: lukethomas
"""

from pathlib import Path
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent          # this script's folder (unet/)
ROOT = HERE.parent                    # the SimAFM-main folder
sys.path.append(str(ROOT))            # so `from unet.model import ...` works from any folder

from unet.model import UNet
from unet.dataset import load_table, split_table_ratio, AFMDenoiseDataset


# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    "run_name": "unet_5prot_tilt30_exact",

    # Train / validation / test split (see split_table_ratio in afm_dataset.py)
    # - train: the network learns from these
    # - val:   checked during training to pick the best epoch (never trained on)
    # - test:  only touched once at the very end; these are the numbers you
    #          report, because nothing was tuned on them
    # Split is by whole groups so near-identical frames can't leak between
    # sets. Now: "clean_path" (protein + tip + cone). With several proteins:
    # "protein", so the test is on molecules the network has never seen.
    "group_by": "protein",
    "ratios": (0.70, 0.15, 0.15),

    # Which orientation modes (the `orientation` column from sweep.yaml) to
    # use. None = everything in data_table.csv. Typical: train on rotated
    # views for variety, evaluate on the realistic fixed orientation:
    #   "train_orientations": ["tilt30"], "eval_orientations": ["fixed"]
    # (eval applies to both val and test). The split above is by protein /
    # stack first, so a protein in test is never seen in ANY orientation.
    "train_orientations": None,
    "eval_orientations": ["fixed"],

    # Model
    "base_channels": 32,    # 64 = his exact network (31 M params); 32 = 7.8 M
    "residual": True,       # predict the noise and subtract it (see unet.py)

    # Training
    "epochs": 50,
    "batch_size": 16,
    "lr": 1e-3,             # Adam learning rate
    "fresh_noise": True,    # new noise every epoch (see afm_dataset.py)
    "augment": True,        # random flips / 90 degree rotations
    "seed": 0,
    "val_every": 1,         # epochs between validation checks
}

DATA_ROOT = ROOT / "data"


# =============================================================================
# Helpers
# =============================================================================
def get_device():
    """Use the GPU if there is one: 'mps' is the Apple Silicon GPU,
    'cuda' an NVIDIA GPU, otherwise fall back to the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def max_clean_height(table, data_root=DATA_ROOT):
    """Largest clean height (nm) in these rows; used as the scale so the
    training images sit roughly in 0-1. Taken from TRAINING rows only, so no
    information about the validation set leaks into training."""
    paths = table["clean_path"].unique()
    return float(max(tifffile.imread(data_root / p).max() for p in paths))


@torch.no_grad()
def predict(model, dataset, device, batch_size=64):
    """Run the model over a dataset; returns (noisy, pred, clean) in nm as
    (N, H, W) numpy arrays."""
    model.eval()                        # eval mode (no dropout etc.)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    noisy_all, pred_all, clean_all = [], [], []
    for noisy, clean, _ in loader:
        pred = model(noisy.to(device)).cpu()
        noisy_all.append(noisy[:, 0].numpy())
        pred_all.append(pred[:, 0].numpy())
        clean_all.append(clean[:, 0].numpy())
    s = dataset.scale                   # undo the scaling -> back to nm
    return (np.concatenate(noisy_all) * s,
            np.concatenate(pred_all) * s,
            np.concatenate(clean_all) * s)


def rmse(a, b):
    """Root-mean-square error per image, in nm. Shape (N, H, W) -> (N,)."""
    return np.sqrt(((a - b) ** 2).mean(axis=(1, 2)))


def best_gaussian(noisy, clean, sigmas=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)):
    """
    Gaussian-filter baseline. Tries several filter widths (in pixels) and
    keeps the one with the lowest RMSE against the clean image. Picking the
    width using the clean image is something you can't do on real data, so
    this is the BEST a Gaussian could possibly do: if the U-Net beats it, the
    comparison is fair (if anything, generous to the Gaussian).
    Returns (filtered images, chosen sigma in pixels).
    """
    best, best_err, best_s = None, np.inf, None
    for s in sigmas:
        filt = np.stack([gaussian_filter(im, s) for im in noisy])
        err = rmse(filt, clean).mean()
        if err < best_err:
            best, best_err, best_s = filt, err, s
    return best, best_s


# =============================================================================
# Main
# =============================================================================
def main(cfg=CONFIG, data_root=DATA_ROOT):
    """
    cfg       : the settings (CONFIG above)
    data_root : folder holding data_table.csv and clean/. Normally data/;
                another folder lets you test on a small throwaway sweep.
    """
    data_root = Path(data_root)
    OUT_DIR = ROOT / "results" / cfg["run_name"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # ---- Data ----
    table = load_table(data_root)
    train_tab, val_tab, test_tab = split_table_ratio(
        table, cfg["group_by"], cfg["ratios"], seed=cfg["seed"])

    # keep only the requested orientation modes in each set
    def keep(t, modes, name):
        if modes is None or "orientation" not in t:
            return t
        out = t[t["orientation"].isin(modes)].reset_index(drop=True)
        if out.empty:
            raise ValueError(f"No {name} rows with orientation in {modes}. "
                             f"Available: {sorted(t['orientation'].unique())}")
        return out
    train_tab = keep(train_tab, cfg.get("train_orientations"), "train")
    val_tab = keep(val_tab, cfg.get("eval_orientations"), "val")
    test_tab = keep(test_tab, cfg.get("eval_orientations"), "test")
    # record which groups went where, so the split is documented and repeatable
    pd.concat([t.assign(split=name) for name, t in
               [("train", train_tab), ("val", val_tab), ("test", test_tab)]]
              ).to_csv(OUT_DIR / "split.csv", index=False)
    for name, t in [("train", train_tab), ("val", val_tab), ("test", test_tab)]:
        print(f"{name:5s}: {len(t) / len(table):5.1%} of rows  "
              f"{sorted(t[cfg['group_by']].unique())}")
    scale = max_clean_height(train_tab, data_root)
    print(f"scale {scale:.3f} nm")

    train_ds = AFMDenoiseDataset(train_tab, data_root, scale,
                                 fresh_noise=cfg["fresh_noise"],
                                 augment=cfg["augment"], seed=cfg["seed"])
    # validation: the saved noisy files, no augmentation -> fixed, repeatable
    val_ds = AFMDenoiseDataset(val_tab, data_root, scale)
    test_ds = AFMDenoiseDataset(test_tab, data_root, scale)
    print(f"Frames: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}")

    # DataLoader batches the samples and shuffles training order each epoch.
    # num_workers=0 keeps loading in the main process (safest in Spyder/macOS).
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # ---- Model, loss, optimiser ----
    model = UNet(in_channels=1, out_channels=1,
                 base_channels=cfg["base_channels"],
                 residual=cfg["residual"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net with {n_params / 1e6:.2f} M parameters")

    # MSE loss: mean of (prediction - clean)^2 over all pixels.
    # This replaces BCEWithLogitsLoss from the segmentation version, which
    # compares predicted probabilities to 0/1 labels. We are predicting
    # continuous heights, so we penalise the squared height error instead.
    # (nn.L1Loss() is a common alternative; it tends to keep edges sharper.)
    loss_fn = nn.MSELoss()

    # Adam adjusts each weight using a running estimate of its gradient;
    # lr sets the step size.
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # Cosine schedule: lr decays smoothly from lr to ~0 over training, so
    # early epochs take big steps and late epochs fine-tune.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=cfg["epochs"])

    # ---- Training loop ----
    # history gets one row per validation check and is saved to history.csv
    # every time, so the loss curve can be plotted (even mid-training) by
    # analysis/make_figures.py without retraining
    history = []
    best_val = np.inf
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()                   # training mode
        lr = optimiser.param_groups[0]["lr"]   # learning rate used this epoch
        running = 0.0
        for noisy, clean, _ in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            pred = model(noisy)                 # 1. forward pass
            loss = loss_fn(pred, clean)         # 2. how wrong is it?
            optimiser.zero_grad()               # 3. clear old gradients
            loss.backward()                     # 4. backprop: d(loss)/d(weights)
            optimiser.step()                    # 5. nudge weights downhill

            running += loss.item() * len(noisy)
        scheduler.step()
        train_loss = running / len(train_ds)

        if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
            model.eval()
            with torch.no_grad():               # no gradients needed to score
                val_loss = sum(loss_fn(model(n.to(device)), c.to(device)).item() * len(n)
                               for n, c, _ in val_loader) / len(val_ds)

            # keep the weights from the epoch with the lowest val loss
            is_best = val_loss < best_val
            if is_best:
                best_val = val_loss
                torch.save({"state_dict": model.state_dict(),
                            "config": cfg, "scale": scale}, OUT_DIR / "best_model.pt")

            # losses are MSE in scaled units; sqrt(MSE) * scale is an RMSE in nm.
            # (train_loss is on fresh noise + augmentation, averaged over the
            # epoch while the weights were changing, so it is only a rough
            # guide; val_loss is on fixed noise with the end-of-epoch weights.)
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse_nm": np.sqrt(train_loss) * scale,
                "val_rmse_nm": np.sqrt(val_loss) * scale,
                "lr": lr,
                "is_best": is_best,
            })
            pd.DataFrame(history).to_csv(OUT_DIR / "history.csv", index=False)
            print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  "
                  f"(val RMSE {np.sqrt(val_loss) * scale:.3f} nm)  "
                  f"{time.time() - t0:.0f}s")

    # ---- Evaluate the best model on the TEST set ----
    # The test set was not used for training or for choosing the epoch,
    # so this is an honest estimate of performance on unseen data.
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    noisy, pred, clean = predict(model, test_ds, device)

    meta = test_tab.iloc[test_ds.row].reset_index(drop=True)  # params per frame
    rows = []
    group_cols = [c for c in ["protein", "orientation", "tip_radius",
                              "cone_angle", "noise"] if c in meta]
    for keys, grp in meta.groupby(group_cols):
        idx = grp.index.to_numpy()
        gauss, g_sigma = best_gaussian(noisy[idx], clean[idx])
        rows.append({
            **dict(zip(group_cols, keys)),
            "rmse_noisy_nm": rmse(noisy[idx], clean[idx]).mean(),
            "rmse_gaussian_nm": rmse(gauss, clean[idx]).mean(),
            "gaussian_sigma_px": g_sigma,
            "rmse_unet_nm": rmse(pred[idx], clean[idx]).mean(),
        })
    metrics = pd.DataFrame(rows)
    metrics["unet_vs_gaussian_%"] = 100 * (1 - metrics["rmse_unet_nm"]
                                           / metrics["rmse_gaussian_nm"])
    metrics.to_csv(OUT_DIR / "metrics.csv", index=False)
    # data_root is saved (relative to SimAFM-main, e.g. "data") so
    # make_figures.py knows where the clean stacks for this run are
    data_root_str = os.path.relpath(data_root, ROOT)
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({**cfg, "scale_nm": scale, "data_root": data_root_str}, f, indent=2)
    pd.set_option("display.width", 200)
    print("\nTest RMSE (nm), lower is better:")
    print(metrics.round(4).to_string(index=False))

    print(f"\nSaved results to {OUT_DIR}")
    return model, metrics


if __name__ == "__main__":
    main()
