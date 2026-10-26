#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim/sweep.py

Generates the clean SimAFM stacks for every protein x clean-parameter
combination in sweep.yaml, and
writes data/data_table.csv with one row per (clean stack, noise level).

Parameters such as tip_radius, cone_angle and orientation can go in either
`clean_sweep` (a list of values to sweep) or `fixed` (one value for all).

For each clean stack two files are written:
- <name>.tif         : the (n_frames, 64, 64) height stack in nm
- <name>_frames.csv  : one row per frame with its ground truth (rotation
                       angles, tilt, x/y jitter, whether it was clipped)

Noise is NOT saved to disk. Each row stores a `noise_seed`;
sim.utils.add_noise(clean, noise, noise_seed) rebuilds exactly the same
noisy stack whenever it is needed (training, testing, or export_noisy.py).

Clean stacks are cached: an existing stack is reused if it has the right
number of frames. The name includes orientation and renderer, so changing
those makes new files rather than reusing old ones. Delete a stack to force
it to be regenerated.

Some proteins are too big for the 64x64 canvas in some views, so a few
frames are cropped at the edge. That is expected, so instead of a warning
per stack, one summary line is printed at the end of the sweep.

Run: F5 in Spyder, or `python sim/sweep.py` from any folder.

@author: lukethomas
"""
import itertools
import sys
from pathlib import Path

import pandas as pd
import tifffile
import yaml

HERE = Path(__file__).parent          # this script's folder (sim/)
ROOT = HERE.parent                    # the SimAFM-main folder
sys.path.append(str(ROOT))            # so `from sim.utils import ...` works from any folder

from sim.simulate import generate_data
from sim.utils import clean_name, noise_name, name_seed, save_tiff

DATA = ROOT / "data"


def expand(sweep):
    """{'a': [1, 2], 'b': [3]} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]"""
    keys = list(sweep)
    return [dict(zip(keys, vals)) for vals in itertools.product(*sweep.values())]


def run_sweep(config_path=HERE / "sweep.yaml", data_root=DATA):
    """
    config_path : the sweep settings (sweep.yaml)
    data_root   : where clean/ and data_table.csv are written. Normally
                  data/; a different folder lets you try a small test
                  sweep without touching the real data.
    """
    data_root = Path(data_root)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    fixed = cfg['fixed']
    renderer = fixed.get('renderer', 'exact')
    clean_combos = expand(cfg["clean_sweep"])
    noise_combos = expand(cfg["noise_sweep"])

    rows = []
    clipped = []       # (stack name, frames clipped, total frames) for the summary
    for protein in cfg["proteins"]:
        for clean_params in clean_combos:

            # Any simulation parameter can be either swept (clean_sweep) or
            # held constant (fixed). Merge them; a swept value wins if a
            # parameter appears in both.
            sim = {**fixed, **clean_params}
            name_params = {'tip_radius': sim['tip_radius'],
                           'cone_angle': sim['cone_angle'],
                           'orientation': sim.get('orientation', 'fixed')}

            clean_filename = clean_name(protein, **name_params, renderer=renderer)
            clean_path = data_root / "clean" / protein / f"{clean_filename}.tif"
            frames_path = clean_path.with_name(f"{clean_filename}_frames.csv")
            crc_seed = name_seed(clean_filename)      # seed for rotations + jitter

            # reuse the cached stack only if it is complete and the right length
            cached = (clean_path.exists() and frames_path.exists()
                      and len(tifffile.imread(clean_path)) == fixed['n_frames'])
            if cached:
                print(f"cached     {clean_filename}")
            else:
                print(f"simulating {clean_filename}")
                clean, frames = generate_data(protein, crc_seed, **sim)
                save_tiff(clean_path, clean)
                frames.to_csv(frames_path, index=False)

            # count frames where the molecule touches the canvas edge
            # (cached stacks: read it back from their _frames.csv)
            if cached:
                frames = pd.read_csv(frames_path)
            n_clip = int(frames['clipped'].sum())
            if n_clip:
                clipped.append((clean_filename, n_clip, len(frames)))

            # one table row per noise level; the noise itself is made later
            # from noise_seed, so nothing is written to disk here
            for noise_params in noise_combos:
                noise_filename = noise_name(protein, **noise_params, **name_params,
                                            renderer=renderer)
                rows.append({
                    'protein': protein,
                    'seed': crc_seed,
                    'noise_seed': name_seed(noise_filename),
                    'clean_path': clean_path.relative_to(data_root).as_posix(),
                    'frames_path': frames_path.relative_to(data_root).as_posix(),
                    'noise_name': noise_filename,
                    'renderer': renderer,
                    **name_params,          # tip_radius, cone_angle, orientation
                    **noise_params,
                })

    data_table = pd.DataFrame(rows).set_index('protein')
    data_table.to_csv(data_root / 'data_table.csv')
    print(f"wrote {len(data_table)} rows to {data_root / 'data_table.csv'}")

    # one summary line about cropping instead of a warning per stack
    n_stacks = len(cfg["proteins"]) * len(clean_combos)
    if clipped:
        names = ", ".join(f"{name} ({n}/{tot})" for name, n, tot in clipped)
        print(f"edge cropping: {len(clipped)}/{n_stacks} stacks have frames "
              f"touching the canvas edge: {names}")
    else:
        print(f"edge cropping: none of the {n_stacks} stacks touch the canvas edge")
    return data_table


if __name__ == "__main__":
    # only runs when you run this file, not when another file imports it
    run_sweep()
