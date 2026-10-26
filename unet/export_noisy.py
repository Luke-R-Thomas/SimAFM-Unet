#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet/export_noisy.py

Rebuilds noisy stacks from the clean tiffs + the seeds in data_table.csv and,
if you ask, saves them as tiffs and/or plots them. Training does NOT need
this; it is for looking at the data or taking noisy images into NanoLocz.

Because the noise comes from a fixed seed (sim.utils.add_noise), an exported
stack is exactly what the U-Net validated / tested on.

Run: edit the call at the bottom and press F5, or
`python unet/export_noisy.py` from any folder.

Examples (in the Spyder console, after running this file once)
--------

    # look at one condition, don't save anything
    export_noisy(protein="6j6j", tip_radius=0.5, cone_angle=1.0, noise=0.5,
                 save=False, show=True)

    # save every noise level for one protein to data/noise/6j6j/
    export_noisy(protein="6j6j")

@author: lukethomas
"""
import sys
from pathlib import Path

import pandas as pd
import tifffile
import yaml

HERE = Path(__file__).parent          # this script's folder (unet/)
ROOT = HERE.parent                    # the SimAFM-main folder
sys.path.append(str(ROOT))            # so `from sim.utils import ...` works from any folder

from sim.utils import add_noise, save_tiff, plot_noisy

DATA = ROOT / "data"


def export_noisy(protein=None, tip_radius=None, cone_angle=None, noise=None,
                 orientation=None,
                 save=True, show=False, frame=0, out_dir=DATA / "noise"):
    """
    Any filter left as None means "all values". Returns {noise_name: stack}.

    save  : write <out_dir>/<protein>/<noise_name>.tif
    show  : plot one frame (`frame`) of each stack
    """
    table = pd.read_csv(DATA / "data_table.csv")
    with open(ROOT / "sim" / "sweep.yaml") as f:
        pix_per_nm = yaml.safe_load(f)["fixed"]["pix_per_nm"]

    # keep only rows matching the filters that were given
    for col, val in [("protein", protein), ("tip_radius", tip_radius),
                     ("cone_angle", cone_angle), ("noise", noise),
                     ("orientation", orientation)]:
        if val is not None:
            table = table[table[col] == val]
    if table.empty:
        raise ValueError("No rows in data_table.csv match those filters.")

    stacks = {}
    for _, row in table.iterrows():
        clean = tifffile.imread(DATA / row["clean_path"])
        noisy = add_noise(clean, row["noise"], row["noise_seed"])
        stacks[row["noise_name"]] = noisy

        if save:
            path = Path(out_dir) / row["protein"] / f"{row['noise_name']}.tif"
            save_tiff(path, noisy)
            print(f"saved {path}")
        if show:
            plot_noisy(noisy[frame], pix_per_nm, row["noise"])
    return stacks


if __name__ == "__main__":
    # edit and run (F5) to export what you want to look at
    export_noisy(protein="6j6j", tip_radius=0.5, cone_angle=1.0, noise=0.5,
                 save=True, show=True)
