#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/make_figures.py

Makes every dissertation figure for one training run, from the files that
unet/train.py saved in results/<run>/ plus the clean stacks it used. It
NEVER retrains: it loads best_model.pt and re-runs it on the test rows from
split.csv. The test noise is rebuilt from fixed seeds, so the inputs are
identical every time.

The U-Net outputs are saved to results/<run>/predictions.npz the first
time, so afterwards restyling a figure and re-running is instant. (Delete
predictions.npz to force the model to be re-run; it is also redone
automatically if best_model.pt is newer.)

Figures (written to results/<run>/figures/ as PDF and 300 dpi PNG):
1. loss_curve          train / val RMSE (nm) vs epoch, best epoch marked
2. rmse_vs_noise       no filter, best Gaussian and U-Net vs noise level
3. examples            noisy | Gaussian | U-Net | clean, plus U-Net - clean
4. line_profile        heights along a line through the molecule
5. improvement_heatmap % improvement of the U-Net over the Gaussian

Run: set RUN_NAME below and press F5, or
`python analysis/make_figures.py` from any folder.

To redo just one figure after F5 (in the Spyder console):
    data = load_run(RUN_NAME)
    fig_examples(data)

@author: lukethomas
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
import torch
import yaml

HERE = Path(__file__).parent          # this script's folder (analysis/)
ROOT = HERE.parent                    # the SimAFM-main folder
sys.path.append(str(ROOT))            # so `from unet.model import ...` works from any folder

from unet.model import UNet
from unet.dataset import AFMDenoiseDataset
from unet.train import predict, rmse, best_gaussian


# =============================================================================
# Settings
# =============================================================================
RUN_NAME = "unet_5prot_tilt30_exact"   # folder in results/
PROFILE_NOISE = 0.5                    # noise level (nm) for the line profile

# pixel size, from the sweep settings (needed to put the image axes in nm)
with open(ROOT / "sim" / "sweep.yaml") as f:
    PIX_PER_NM = yaml.safe_load(f)["fixed"]["pix_per_nm"]


# =============================================================================
# Style: one place that sets the look of every figure
# =============================================================================
FULL_WIDTH = 6.3     # inches: text width of an A4 page with ~2.5 cm margins

# Okabe-Ito colours: distinguishable with every common type of colour blindness
COLOURS = {
    "clean": "#000000",      # black
    "noisy": "#999999",      # grey
    "gaussian": "#E69F00",   # orange
    "unet": "#0072B2",       # blue
    "train": "#E69F00",
    "val": "#0072B2",
}
AFM_CMAP = ListedColormap(np.load(ROOT / "AFM_cmap.npy"))   # for height images
DIFF_CMAP = "RdBu_r"   # diverging (blue = too low, red = too high) for errors


def set_style():
    """Fonts, grid and colours used by every figure."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.grid": True,             # light grid on line plots
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,            # real text in the PDF (editable, searchable)
    })


def save_fig(fig, data, name):
    """Save as vector PDF (for the dissertation) and 300 dpi PNG."""
    out = data["run_dir"] / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    print(f"saved figures/{name}.pdf and .png")


def show_height(ax, img, vmin, vmax, cmap=AFM_CMAP):
    """An image with its axes in nm and no grid."""
    extent = [0, img.shape[1] / PIX_PER_NM, img.shape[0] / PIX_PER_NM, 0]
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.grid(False)
    return im


# =============================================================================
# Loading the run (and the cached predictions)
# =============================================================================
def load_run(run):
    """
    Everything the figures need, in one dict:
    run_dir, config, history (None if there is no history.csv), and, if the
    clean stacks are available, the test-set images and their settings:
      noisy, gauss, pred, clean : (N, H, W) arrays in nm, one per test frame
      meta                      : DataFrame, one row per frame (protein,
                                  tip_radius, noise, ...)
    """
    run_dir = ROOT / "results" / run
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    # older runs didn't record it; they used data/
    data_root = ROOT / config.get("data_root", "data")

    history = None
    if (run_dir / "history.csv").exists():
        history = pd.read_csv(run_dir / "history.csv")

    data = {"run": run, "run_dir": run_dir, "config": config,
            "history": history, "data_root": data_root}

    # test rows, in the same order train.py used
    split = pd.read_csv(run_dir / "split.csv")
    test_tab = split[split["split"] == "test"].reset_index(drop=True)

    # check the clean stacks are there before trying to use them
    missing = [p for p in test_tab["clean_path"].unique()
               if not (data_root / p).exists()]
    if missing:
        print(f"\n{len(missing)} clean stack(s) used by the test set are missing "
              f"from {data_root}:")
        for p in missing:
            print(f"  {p}")
        print("Run sim/sweep.py with the settings this run used to make them. "
              "Only figures that don't need them can be made.")
        data["missing"] = missing
        return data

    data.update(get_predictions(run_dir, test_tab, data_root))
    return data


def get_predictions(run_dir, test_tab, data_root):
    """
    Test-set noisy / Gaussian / U-Net / clean images. Loaded from
    predictions.npz if it is up to date, otherwise made (on the CPU) and saved.
    """
    cache = run_dir / "predictions.npz"
    model_file = run_dir / "best_model.pt"

    if cache.exists() and cache.stat().st_mtime > model_file.stat().st_mtime:
        print("loading saved predictions.npz")
        npz = np.load(cache)
        # the per-frame settings were saved as columns named meta_<column>
        meta = pd.DataFrame({k[5:]: npz[k] for k in npz.files if k.startswith("meta_")})
        return {"noisy": npz["noisy"], "gauss": npz["gauss"],
                "pred": npz["pred"], "clean": npz["clean"], "meta": meta}

    print("running the U-Net on the test set (CPU)...")
    # weights_only=False: the file also holds the config dict; it is our own file
    ckpt = torch.load(model_file, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = UNet(in_channels=1, out_channels=1,
                 base_channels=cfg["base_channels"], residual=cfg["residual"])
    model.load_state_dict(ckpt["state_dict"])

    # fixed, seeded noise (fresh_noise=False), same scale as training
    test_ds = AFMDenoiseDataset(test_tab, data_root, ckpt["scale"])
    noisy, pred, clean = predict(model, test_ds, torch.device("cpu"))

    # settings for every frame, and which frame of its stack it is
    meta = test_tab.iloc[test_ds.row].reset_index(drop=True)
    meta["frame"] = meta.groupby("noise_name").cumcount()

    # best Gaussian, chosen per group exactly as in train.py
    gauss = np.zeros_like(noisy)
    meta["gaussian_sigma_px"] = 0.0
    group_cols = [c for c in ["protein", "orientation", "tip_radius",
                              "cone_angle", "noise"] if c in meta]
    for _, grp in meta.groupby(group_cols):
        idx = grp.index.to_numpy()
        gauss[idx], sigma = best_gaussian(noisy[idx], clean[idx])
        meta.loc[idx, "gaussian_sigma_px"] = sigma

    # text columns (protein, paths...) are saved as plain strings, not Python
    # objects, so the file loads without pickle
    np.savez_compressed(cache, noisy=noisy, gauss=gauss, pred=pred, clean=clean,
                        **{f"meta_{c}": meta[c].to_numpy(dtype=str if meta[c].dtype == object else None)
                           for c in meta.columns})
    print(f"saved {cache.name}")
    return {"noisy": noisy, "gauss": gauss, "pred": pred, "clean": clean, "meta": meta}


def panels(meta):
    """
    Split the test frames into panels by orientation and cone angle, but only
    for those that have more than one value. Returns [(title, frame mask)].
    """
    cols = [c for c in ["orientation", "cone_angle"]
            if c in meta and meta[c].nunique() > 1]
    if not cols:
        return [("", np.ones(len(meta), bool))]
    out = []
    for keys, grp in meta.groupby(cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        title = ", ".join(f"{k}" if c == "orientation" else f"cone {k:g}°"
                          for c, k in zip(cols, keys))
        out.append((title, meta.index.isin(grp.index)))
    return out


def example_frames(data):
    """
    Indices of the frames used in the example figures: the same molecule and
    frame (the first test stack, frame 0) at each noise level.
    Returns {noise level: index}.
    """
    meta = data["meta"]
    first = meta.iloc[0]
    same = (meta["clean_path"] == first["clean_path"]) & (meta["frame"] == 0)
    return {row["noise"]: i for i, row in meta[same].sort_values("noise").iterrows()}


# =============================================================================
# Figures (one function each)
# =============================================================================
def fig_loss_curve(data):
    """1. Train and validation RMSE (nm) per epoch, best epoch marked."""
    h = data["history"]
    if h is None:
        print("loss_curve skipped: no history.csv in this run "
              "(runs from before history.csv was added don't have one)")
        return None

    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.6, 2.6))
    ax.plot(h["epoch"], h["train_rmse_nm"], color=COLOURS["train"], label="train")
    ax.plot(h["epoch"], h["val_rmse_nm"], color=COLOURS["val"], label="validation")

    # best epoch = lowest validation loss (the saved model)
    best = h.loc[h["val_loss"].idxmin()]
    ax.plot(best["epoch"], best["val_rmse_nm"], "o", color=COLOURS["val"],
            markerfacecolor="white")
    ax.annotate(f"best: epoch {best['epoch']:.0f}\n{best['val_rmse_nm']:.3f} nm",
                (best["epoch"], best["val_rmse_nm"]),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   # whole epochs only
    ax.set_ylabel("RMSE (nm)")
    ax.legend()
    save_fig(fig, data, "loss_curve")
    return fig


def fig_rmse_vs_noise(data):
    """2. RMSE vs noise level for no filter, best Gaussian and U-Net.
    One panel per tip radius (rows for orientation / cone angle if several).
    Points are the mean over test frames, error bars one standard deviation."""
    meta = data["meta"]
    # RMSE of every test frame against its clean image
    err = {"no filter (y = x)": (rmse(data["noisy"], data["clean"]), COLOURS["noisy"]),
           "best Gaussian": (rmse(data["gauss"], data["clean"]), COLOURS["gaussian"]),
           "U-Net": (rmse(data["pred"], data["clean"]), COLOURS["unet"])}

    tips = sorted(meta["tip_radius"].unique())
    rows = panels(meta)
    fig, axes = plt.subplots(len(rows), len(tips), sharex=True, sharey=True,
                             figsize=(FULL_WIDTH, 2.2 * len(rows) + 0.3), squeeze=False)

    for r, (row_title, row_mask) in enumerate(rows):
        for c, tip in enumerate(tips):
            ax = axes[r, c]
            mask = row_mask & (meta["tip_radius"] == tip).to_numpy()
            for label, (values, colour) in err.items():
                df = pd.DataFrame({"noise": meta["noise"][mask], "rmse": values[mask]})
                stats = df.groupby("noise")["rmse"].agg(["mean", "std"])
                ax.errorbar(stats.index, stats["mean"], yerr=stats["std"],
                            marker="o", capsize=2, color=colour, label=label)
            ax.set_title(f"tip radius {tip:g} nm" + (f"\n{row_title}" if row_title else ""))
            if r == len(rows) - 1:
                ax.set_xlabel("Noise level (nm rms)")
            if c == 0:
                ax.set_ylabel("RMSE (nm)")

    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    save_fig(fig, data, "rmse_vs_noise")
    return fig


def fig_examples(data):
    """3. noisy | best Gaussian | U-Net | clean for each noise level (same
    molecule and frame), each row on one colour scale with its own colourbar,
    plus a last row of U-Net minus clean on a symmetric diverging scale."""
    frames = example_frames(data)
    levels = list(frames)
    n_cols = max(4, len(levels))
    fig, axes = plt.subplots(len(levels) + 1, n_cols,
                             figsize=(FULL_WIDTH, 1.45 * (len(levels) + 1)),
                             squeeze=False, layout="constrained")

    titles = ["Noisy", "Best Gaussian", "U-Net", "Clean"]
    for r, lvl in enumerate(levels):
        i = frames[lvl]
        images = [data["noisy"][i], data["gauss"][i], data["pred"][i], data["clean"][i]]
        # shared colour scale for the row, set by the clean image
        vmin, vmax = data["clean"][i].min(), data["clean"][i].max()
        for c, img in enumerate(images):
            im = show_height(axes[r, c], img, vmin, vmax)
            if r == 0:
                axes[r, c].set_title(titles[c])
        axes[r, 0].set_ylabel(f"{lvl:g} nm noise\ny (nm)")
        fig.colorbar(im, ax=axes[r, :], label="Height (nm)", shrink=0.9)

    # last row: U-Net error at each noise level
    r = len(levels)
    diffs = [data["pred"][frames[lvl]] - data["clean"][frames[lvl]] for lvl in levels]
    lim = max(np.abs(d).max() for d in diffs)          # symmetric limits
    for c, (lvl, d) in enumerate(zip(levels, diffs)):
        im = show_height(axes[r, c], d, -lim, lim, cmap=DIFF_CMAP)
        axes[r, c].set_title(f"{lvl:g} nm noise")
        axes[r, c].set_xlabel("x (nm)")
    axes[r, 0].set_ylabel("U-Net − clean\ny (nm)")
    fig.colorbar(im, ax=axes[r, :], label="Error (nm)", shrink=0.9)

    # hide unused panels, and tick labels except on the left column and bottom row
    for ax in axes.flat:
        if not ax.images:
            ax.set_visible(False)
    for ax in axes[:, 1:].flat:
        ax.set_yticklabels([])
    for ax in axes[:-1, :].flat:
        ax.set_xticklabels([])

    save_fig(fig, data, "examples")
    return fig


def fig_line_profile(data, noise=PROFILE_NOISE):
    """4. Heights along a horizontal line through the molecule's centre for
    clean, noisy, Gaussian and U-Net, at one noise level (the nearest one
    available to `noise`)."""
    frames = example_frames(data)
    lvl = min(frames, key=lambda n: abs(n - noise))
    i = frames[lvl]
    clean = data["clean"][i]

    # centre = height-weighted mean row, so the line crosses the molecule
    rows = np.arange(clean.shape[0])
    centre = int(round((clean.sum(axis=1) * rows).sum() / clean.sum()))
    y_nm = (centre + 0.5) / PIX_PER_NM
    x_nm = (np.arange(clean.shape[1]) + 0.5) / PIX_PER_NM

    fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.4),
                                     gridspec_kw={"width_ratios": [1, 2]})
    im = show_height(ax_img, clean, clean.min(), clean.max())
    ax_img.axhline(y_nm, color="white", linestyle="--", linewidth=1)
    ax_img.set_title("Clean, with profile line")
    ax_img.set_xlabel("x (nm)")
    ax_img.set_ylabel("y (nm)")
    fig.colorbar(im, ax=ax_img, label="Height (nm)", shrink=0.8)

    for key, label in [("noisy", "noisy"), ("gauss", "best Gaussian"),
                       ("pred", "U-Net"), ("clean", "clean")]:
        colour = COLOURS["gaussian" if key == "gauss" else
                         "unet" if key == "pred" else key]
        ax.plot(x_nm, data[key][i][centre], color=colour, label=label,
                linewidth=1.6 if key == "clean" else 1.0)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Height (nm)")
    ax.set_title(f"Line profile, {lvl:g} nm noise")
    ax.legend(ncol=2)
    fig.tight_layout()
    save_fig(fig, data, "line_profile")
    return fig


def fig_improvement_heatmap(data):
    """5. % lower RMSE for the U-Net than the best Gaussian, for each tip
    radius x noise level (mean RMSE over test frames in each cell).
    One panel per orientation / cone angle if there are several."""
    meta = data["meta"]
    df = meta[["tip_radius", "noise"]].copy()
    df["unet"] = rmse(data["pred"], data["clean"])
    df["gauss"] = rmse(data["gauss"], data["clean"])

    groups = panels(meta)
    fig, axes = plt.subplots(1, len(groups), squeeze=False,
                             figsize=(FULL_WIDTH * min(1, 0.5 * len(groups) + 0.1), 2.4),
                             layout="constrained")

    # one table per panel: rows = tip radius, columns = noise level
    tables = []
    for _, mask in groups:
        m = df[mask].groupby(["tip_radius", "noise"])[["unet", "gauss"]].mean()
        tables.append((100 * (1 - m["unet"] / m["gauss"])).unstack("noise"))
    lim = max(np.nanmax(np.abs(t.to_numpy())) for t in tables)   # symmetric colours

    for ax, (title, _), t in zip(axes[0], groups, tables):
        im = ax.imshow(t.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(t.shape[1]), [f"{v:g}" for v in t.columns])
        ax.set_yticks(range(t.shape[0]), [f"{v:g}" for v in t.index])
        ax.set_xlabel("Noise level (nm rms)")
        ax.set_title(title)
        # write the number in each cell
        for (r, c), v in np.ndenumerate(t.to_numpy()):
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * lim else "black")
    axes[0, 0].set_ylabel("Tip radius (nm)")
    fig.colorbar(im, ax=axes[0, :], label="Improvement (%)")
    fig.suptitle("U-Net RMSE improvement over the best Gaussian")
    save_fig(fig, data, "improvement_heatmap")
    return fig


# =============================================================================
# Main
# =============================================================================
def main(run=RUN_NAME):
    set_style()
    print(f"Run: {run}")
    data = load_run(run)

    fig_loss_curve(data)
    if "missing" in data:
        print("Stopped: the other figures need the clean stacks listed above.")
        return data

    fig_rmse_vs_noise(data)
    fig_examples(data)
    fig_line_profile(data)
    fig_improvement_heatmap(data)
    print(f"\nFigures are in {data['run_dir'] / 'figures'}")
    return data


if __name__ == "__main__":
    data = main()
