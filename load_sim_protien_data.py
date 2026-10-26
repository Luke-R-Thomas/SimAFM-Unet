#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_sim_protien_data.py

Simulates AFM height images of a protein from its PDB structure.

Pipeline for one frame
----------------------
1. load_coords      : atom positions (nm) from pdb_files/<PDB_ID>.pdb
2. base_orient      : the "natural" orientation (theta_x/y/z, e.g. 90 deg
                      about x), as in the original SimAFM
3. sample_rotation  : an EXTRA random rotation, set by `orientation`:
                        fixed     no extra rotation (LAFM evaluation movies)
                        spin<D>   in-plane rotation, uniform in +/- D deg
                        tilt<D>   tilted up to D deg from the natural
                                  orientation, then spun 0-360 deg in-plane
                        random    any orientation, uniform over the sphere
4. z_cut            : drop the lowest z_thresh fraction (membrane-embedded
                      part), then set the lowest remaining atom to z = 0
5. jitter           : shift x, y by a normal random amount (sd jitter_nm)
6. render_frame     : tip-shape height image on a size x size canvas

Every frame's rotation and jitter is returned so it can be saved and used as
ground truth (e.g. for localisation accuracy).

Original SimAFM code: pymte, 2023. Restructured, vectorised and extended.

@author: lukethomas
"""

import re
from pathlib import Path
from time import time
import warnings

import mdtraj as md
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# folder this file lives in, so pdb_files/ is found wherever you run from
HERE = Path(__file__).parent


# =============================================================================
# Coordinates and orientation
# =============================================================================
def load_coords(PDB_ID):
    """Atom coordinates (N, 3) in nm from pdb_files/<PDB_ID>.pdb."""
    pdb_file = str(HERE / 'pdb_files' / f'{PDB_ID}.pdb')
    start = time()
    protein = md.load(pdb_file)          # mdtraj gives positions in nm
    coords = protein.xyz[0, :, :]        # given as 1xNx3; only want Nx3
    print(f'Time to collect atomic coords: {time() - start:.2f}s')
    return coords


def base_orient(coords, theta_x, theta_y, theta_z):
    """
    The original SimAFM rotations (degrees) about x, then y, then z.
    This sets the protein's natural orientation on the surface.
    """
    coords = coords.copy()

    if theta_x != 0:   # != 0 so negative angles work too
        a = np.deg2rad(theta_x)
        rx = coords.copy()
        rx[:, 1] = coords[:, 1] * np.cos(a) - coords[:, 2] * np.sin(a)
        rx[:, 2] = coords[:, 1] * np.sin(a) + coords[:, 2] * np.cos(a)
        coords[:, 1:3] = rx[:, 1:3]

    if theta_y != 0:
        a = np.deg2rad(theta_y)
        ry = coords.copy()
        ry[:, 0] = coords[:, 0] * np.cos(a) + coords[:, 2] * np.sin(a)
        ry[:, 2] = coords[:, 2] * np.cos(a) - coords[:, 0] * np.sin(a)
        coords[:, 0] = ry[:, 0]
        coords[:, 2] = ry[:, 2]

    if theta_z != 0:
        a = np.deg2rad(theta_z)
        v = np.vstack((coords[:, 0], coords[:, 1]))
        R = np.array([[np.cos(a), np.sin(a)],
                      [-np.sin(a), np.cos(a)]])
        so = np.dot(R, v)
        coords[:, 0] = so[0, :]
        coords[:, 1] = so[1, :]

    return coords


def z_cut(coords, z_thresh):
    """
    Remove the lowest `z_thresh` fraction of the height range (e.g. the part
    embedded in a membrane, which the tip can't reach), then shift z so the
    lowest remaining atom is at 0. z_thresh = 0 keeps the whole molecule.
    """
    if z_thresh > 0:
        cut = z_thresh * (np.max(coords[:, 2]) - np.min(coords[:, 2])) + np.min(coords[:, 2])
        coords = coords[coords[:, 2] > cut]
    coords = coords.copy()
    coords[:, 2] = coords[:, 2] - np.min(coords[:, 2])
    return coords


def prepare_coords(PDB_ID, theta_x, theta_y, theta_z, z_thresh):
    """Original single-orientation version: load, orient, cut. (N, 3) nm."""
    return z_cut(base_orient(load_coords(PDB_ID), theta_x, theta_y, theta_z), z_thresh)


def sample_rotation(orientation, rng):
    """
    The extra rotation for one frame, applied after the natural orientation.

    orientation : 'fixed' | 'spin<D>' | 'tilt<D>' | 'random'  (D in degrees)
    rng         : numpy Generator, so rotations are reproducible from a seed

    Returns a scipy Rotation, or None for 'fixed' (no rotation at all).

    Note on 'tilt': the tilt angle t is drawn so that cos(t) is uniform, not
    t itself. That spreads tilts evenly over the spherical cap; drawing t
    uniformly would over-represent near-upright orientations. 'random' uses
    Rotation.random for the same reason (independent uniform Euler angles
    bunch up near the poles).
    """
    if orientation == 'fixed':
        return None              # no extra rotation

    if orientation == 'random':
        return Rotation.random(random_state=rng)

    m = re.fullmatch(r'(spin|tilt)(\d+(?:\.\d+)?)', str(orientation))
    if m is None:
        raise ValueError(f"orientation '{orientation}' not understood; use "
                         "'fixed', 'spin<deg>', 'tilt<deg>' or 'random'")
    kind, D = m.group(1), float(m.group(2))

    if kind == 'spin':
        # in-plane rotation only (about the vertical axis)
        return Rotation.from_euler('z', rng.uniform(-D, D), degrees=True)

    # tilt: lean over by t towards a random direction phi, then spin in-plane
    phi = rng.uniform(0, 2 * np.pi)
    t = np.arccos(rng.uniform(np.cos(np.deg2rad(D)), 1.0))
    tilt = Rotation.from_rotvec(t * np.array([np.cos(phi), np.sin(phi), 0.0]))
    spin = Rotation.from_euler('z', rng.uniform(0, 360), degrees=True)
    return spin * tilt          # scipy: (a * b) applies b first, then a


def apply_rotation(coords, rotation):
    """Rotate about the molecule's centre so it stays in place."""
    if rotation is None:
        return coords.copy()     # 'fixed': leave coordinates bit-for-bit as is
    centre = coords.mean(axis=0)
    return rotation.apply(coords - centre) + centre


# =============================================================================
# Canvas and rendering
# =============================================================================
def make_canvas(coords, radius, angle, pix_per_nm, size=64, processing=True):
    """
    Where the image sits (origin, in pixels) and how big it is (shape).
    rs     : tip radius in pixels
    fspace : extra margin for the cone flanks, which widen the footprint
    processing=True gives a size x size canvas centred on the molecule
    (what the U-Net uses); False gives a canvas fitted to the molecule.
    """
    rs = radius * pix_per_nm
    fspace = (np.max(coords[:, 2]) - radius) * np.tan(angle * np.pi / 180) * pix_per_nm + 1

    x_min = int(np.floor(min(coords[:, 0] * pix_per_nm) - rs - fspace))
    x_max = int(np.ceil(max(coords[:, 0] * pix_per_nm) + rs + fspace))
    x_centre = (x_max + x_min) / 2

    y_min = int(np.floor(min(coords[:, 1] * pix_per_nm) - rs - fspace))
    y_max = int(np.ceil(max(coords[:, 1] * pix_per_nm) + rs + fspace))
    y_centre = (y_max + y_min) / 2

    if processing is True:
        origin = ((round(x_centre) - size // 2), (round(y_centre) - size // 2))
        shape = (size, size)
    else:
        origin = ([x_min, y_min])
        shape = ([x_max - x_min + 1, y_max - y_min + 1])

    return origin, shape, rs, fspace


def _tip_height(z, d2, radius, tan_a):
    """
    Height of the tip apex when touching an atom at height z, for a pixel at
    squared in-plane distance d2 (nm^2) from the atom.
    - spherical apex, d < r :  h = z + sqrt(r^2 - d^2) - r
    - cone side,     d >= r :  h = z - r - (d - r) / tan(a)
    """
    dh = np.sqrt(np.maximum(radius ** 2 - d2, 0)) - radius
    di = np.sqrt(d2) - radius
    return np.where(dh > -radius, z + dh, z - radius - di / tan_a)


def render_frame(coords, origin, shape, rs, fspace, radius, angle, pix_per_nm,
                 renderer='exact', chunk=20000):
    """
    Simulated AFM height image: for every pixel, the highest point the tip
    touches on any atom (max over atoms of the tip shape), on a pixel grid
    starting at `origin`.

    renderer
    --------
    'exact'    : the height at each pixel is computed at that pixel's true
                 distance from the atom. Sub-pixel shifts of the molecule move
                 the image correctly (centroid error ~0.001 nm in testing).
    'original' : the original SimAFM placement, reproduced exactly. Heights
                 are computed on an offset grid then rounded into pixels, up
                 to half a pixel from where they were computed, so sub-pixel
                 shifts are distorted (centroid error ~0.02 nm rms, about 20%
                 of a 0.1 nm jitter). Kept for comparison with SimAFM.

    Both are vectorised (all atoms x neighbourhood pixels as arrays).
    """
    img = np.zeros(shape)

    # x, y in pixels; z stays in nm. (Multiplied before converting to
    # float64, as the original did, so 'original' reproduces it bit for bit.)
    xs = (coords[:, 0] * pix_per_nm).astype(np.float64)
    ys = (coords[:, 1] * pix_per_nm).astype(np.float64)
    zs = coords[:, 2].astype(np.float64)
    tan_a = np.tan(angle * np.pi / 180)

    if renderer == 'exact':
        # integer pixel offsets covering the tip footprint
        R = int(np.ceil(rs + fspace))
        d = np.arange(-R, R + 1)
    elif renderer == 'original':
        d = np.arange(-rs - fspace, rs + fspace)
    else:
        raise ValueError("renderer must be 'exact' or 'original'")
    dx, dy = np.meshgrid(d, d, indexing='ij')
    dx, dy = dx.ravel('F'), dy.ravel('F')                     # (G,)

    # process atoms in chunks so very large proteins don't run out of memory
    for c in range(0, len(xs), chunk):
        x, y, z = xs[c:c + chunk, None], ys[c:c + chunk, None], zs[c:c + chunk, None]

        if renderer == 'exact':
            # real pixels around the atom's nearest pixel           (N, G)
            px = np.round(x) + dx[None, :]
            py = np.round(y) + dy[None, :]
            # true distance from each pixel centre to the atom, in nm
            d2 = ((px - x) / pix_per_nm) ** 2 + ((py - y) / pix_per_nm) ** 2
            h = _tip_height(z, d2, radius, tan_a)
            pos_x = (px - origin[0]).astype(int)
            pos_y = (py - origin[1]).astype(int)
        else:
            # original SimAFM placement
            dxl = dx[None, :] - (np.round(x) - x)
            dyl = dy[None, :] - (np.round(y) - y)
            d2 = (dxl / pix_per_nm) ** 2 + (dyl / pix_per_nm) ** 2
            # (same as _tip_height but in the original's operation order)
            dh = np.sqrt(np.maximum(-(dxl / pix_per_nm) ** 2 - (dyl / pix_per_nm) ** 2
                                    + radius ** 2, 0)) - radius
            di = np.sqrt(d2) - radius
            h = np.where(dh > -radius, z + dh, z - radius - (di / tan_a))
            pos_x = np.round(x + dxl - origin[0]).astype(int)
            pos_y = np.round(y + dyl - origin[1]).astype(int)

        # keep pixels inside the canvas; for each pixel keep the highest value
        ok = (pos_x >= 0) & (pos_x < shape[0]) & (pos_y >= 0) & (pos_y < shape[1])
        np.maximum.at(img, (pos_x[ok], pos_y[ok]), h[ok])

    return img


# =============================================================================
# Stack generation
# =============================================================================
def generate_data(PDB_ID, crc_seed, cone_angle, tip_radius, pix_per_nm, n_frames,
                  orientation='fixed', renderer='exact', z_thresh=0.5,
                  theta_x=90.0, theta_y=0.0, theta_z=0.0, jitter_nm=0.1,
                  size=64):
    """
    Simulate a stack of n_frames AFM images of one protein.

    Each frame: natural orientation -> extra rotation (from `orientation`)
    -> z cut -> centre the size x size canvas on the molecule -> jitter x, y
    -> render. In 'fixed' mode the rotation is the same every frame, so the
    frames differ only by jitter (the LAFM movie); in the other modes every
    frame is a different view of the protein.

    Returns
    -------
    stack  : (n_frames, size, size) heights in nm
    frames : DataFrame, one row per frame, with the ground truth:
             rot_z1, rot_x, rot_z2 : the extra rotation as ZXZ Euler angles (deg)
             tilt_deg              : how far the molecule is tilted from its
                                     natural orientation
             jitter_x_nm, jitter_y_nm : the applied jitter
             clipped               : True if the molecule touched the canvas
                                     edge (canvas too small for this view)
    """
    rng = np.random.default_rng(crc_seed)

    natural = base_orient(load_coords(PDB_ID), theta_x, theta_y, theta_z)

    stack = np.zeros((n_frames, size, size))
    records = []

    for n in range(n_frames):
        rot = sample_rotation(orientation, rng)
        coords = z_cut(apply_rotation(natural, rot), z_thresh)

        # canvas centred on this view of the molecule, BEFORE jitter, so the
        # jitter shows up as a real shift of the molecule in the image
        origin, shape, rs, fspace = make_canvas(coords, tip_radius, cone_angle,
                                                pix_per_nm, size=size, processing=True)

        # gives the x and y shifts by a normal distribution (sd jitter_nm)
        jitter_shift = rng.normal(0, jitter_nm, size=2)
        working_coords = coords.copy()
        working_coords[:, :2] += jitter_shift

        img = render_frame(working_coords, origin, shape, rs, fspace,
                           tip_radius, cone_angle, pix_per_nm, renderer=renderer)
        stack[n] = img

        # any height on the border means part of the molecule was cut off
        border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
        if rot is None:
            rot = Rotation.identity()            # for the record only
        z1, x, z2 = rot.as_euler('ZXZ', degrees=True)
        records.append({
            'frame': n,
            'rot_z1': z1, 'rot_x': x, 'rot_z2': z2,
            'tilt_deg': np.degrees(np.arccos(np.clip(rot.apply([0, 0, 1])[2], -1, 1))),
            'jitter_x_nm': jitter_shift[0], 'jitter_y_nm': jitter_shift[1],
            'clipped': bool((border > 0).any()),
        })

    frames = pd.DataFrame(records)
    n_clip = int(frames['clipped'].sum())
    if n_clip:
        warnings.warn(f"{PDB_ID} tr={tip_radius} ca={cone_angle} {orientation}: "
                      f"{n_clip}/{n_frames} frames touch the {size}x{size} canvas "
                      f"edge, so part of the molecule is cut off. Use a bigger canvas.")
    return stack, frames
