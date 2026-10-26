#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:30:19 2026

@author: lukethomas
"""

import itertools
from pathlib import Path
from tools import plot_clean, plot_noisy, clean_name, noise_name, name_seed, add_noise
import numpy as np
import tifffile
import yaml


HERE = Path(__file__).parent
ROOT = HERE / "data"

with open(HERE / "sweep.yaml") as f:
    cfg = yaml.safe_load(f)

def expand(sweep):
    keys = list(sweep)
    return [dict(zip(keys, vals)) for vals in itertools.product(*sweep.values())]


# ---- settings ----
protein = "6j6j"
combo_idx = 3          # which clean combination to inspect
frame = 0              # which frame to compare clean vs noisy
cmap = "AFM_cmap.npy"
ppn = cfg["fixed"]["pix_per_nm"]

clean_combos = expand(cfg["clean_sweep"])
noise_combos = expand(cfg["noise_sweep"])

print("clean combos:")
for i, c in enumerate(clean_combos):
    print(f"  {i}: {c}")

# swept values override fixed ones (same rule as sweep.py)
sim = {**cfg["fixed"], **clean_combos[combo_idx]}
clean_params = {"tip_radius": sim["tip_radius"], "cone_angle": sim["cone_angle"],
                "orientation": sim.get("orientation", "fixed")}
print(f"\ninspecting {protein} {clean_params}")

# ---- load clean ----
renderer = cfg["fixed"].get("renderer", "exact")
clean_path = ROOT / "clean" / protein / f"{clean_name(protein, **clean_params, renderer=renderer)}.tif"
clean = tifffile.imread(clean_path)
print(f"clean: shape {clean.shape}, dtype {clean.dtype}, "
      f"height {clean.min():.3f} to {clean.max():.3f}")

# fixed colour limits from the clean frame, used for every plot
vmin, vmax = np.percentile(clean[frame], [1, 99])

# ---- a few frames of the clean stack ----
for i in np.linspace(0, clean.shape[0] - 1, 4, dtype=int):
    plot_clean(clean[i], ppn, vmin=vmin, vmax=vmax, cmap_path=cmap, figsize=(6, 6))

# ---- noisy vs clean ----
plot_clean(clean[frame], ppn, vmin=vmin, vmax=vmax, cmap_path=cmap, figsize=(6, 6))

for noise_params in noise_combos:
    sigma = noise_params["noise"]

    # rebuild the seeded noise exactly as training / export_noisy.py do
    # (noisy tiffs are no longer saved by sweep.py)
    noisy_name = noise_name(protein, sigma, **clean_params, renderer=renderer)
    noisy = add_noise(clean, sigma, name_seed(noisy_name))

    assert noisy.shape == clean.shape, f"shape mismatch at sigma={sigma}"

    resid = noisy.astype(np.float64) - clean.astype(np.float64)
    ratio = resid.std() / sigma
    print(f"sigma={sigma}: dtype {noisy.dtype}, residual mean {resid.mean():+.4f}, "
          f"std {resid.std():.4f} (ratio {ratio:.3f}), "
          f"SNR {np.ptp(clean[frame]) / sigma:.1f}")

    if abs(ratio - 1) > 0.05:
        print("WARNING: residual std doesn't match sigma, check noise scaling or units")

    plot_noisy(noisy[frame], ppn, sigma, vmin=vmin, vmax=vmax, cmap_path=cmap, figsize=(6, 6))



























# ============================================================================= 
# place = np.zeros(((3,256,256)))
# 
# # Example: 3 separate 2D image arrays (Height, Width)
# img1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
# img2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
# img3 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
# 
# # Stack them along a new axis to create a 3D array (Frames, Height, Width)
# combined_array = np.stack([img1, img2, img3], axis=0)
# 
# # Save everything into one multi-page TIFF file
# tifffile.imwrite('single_multipage_file.tif', combined_array)
# 
# 
# 
# here = Path(__file__).parent          # use Path.cwd() in a notebook
# stack = tifffile.imread(here / 'single_multipage_file.tif')
# 
# print(stack.shape)   # e.g. (100, 512, 512)
# 
# first = stack[0]     # frame 1
# tenth = stack[1]     # frame 10
# last = stack[2]     # frame 100 (or stack[-1])
# 
# print(first.shape)   # (512, 512), a single image
# 
# place[0] = first
# =============================================================================



# =============================================================================
# seed = 43
# 
# rng = np.random.default_rng(seed)
# 
# # gives x and y shifts by a normal distributiion +/- 0.1nm 
# 
# 
# a =rng.normal(0, 0.1, size=2)  
# 
# # (N, 3) in nm
# 
# # Atom 1 x y z
# # Atom 2 x y z
# 
#        # 2 numbers from a normal distribution, mean 0, std 0.1
# b = rng.normal(0, 0.1, size=(64,64))  # a 64 x 64 array, e.g. noise for one image
# c = rng.uniform(0.05, 0.3)            # 1 number evenly spread between 0.05 and 0.3
# d = rng.integers(0, 10)               # a whole number from 0 to 9
# =============================================================================
