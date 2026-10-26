#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim/utils.py  (was tools.py)

Small helpers shared by the simulation, the U-Net and the plotting scripts:
- clean_name, noise_name : the filenames for each stack, e.g.
                           6j6j_tr05_ca10_tilt30_exact(_n05)
- name_seed              : turns a filename into a fixed random seed
- add_noise              : the ONE place Gaussian noise is added
- save_tiff              : writes a stack as a float32 tiff
- show, plot_clean, plot_noisy : quick looks at single images

Do NOT change the naming or seed functions: every seed (jitter, rotations,
noise) comes from the filenames, so changing them changes all the data.

@author: lukethomas
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import tifffile
import zlib

HERE = Path(__file__).parent          # this file's folder (sim/)
ROOT = HERE.parent                    # the SimAFM-main folder, where AFM_cmap.npy is



def show(img,pix_per_nm,title, vmin, vmax, cmap_path, figuresize):
    
    AFM = ListedColormap(np.load(ROOT / cmap_path))
    extent = [0, img.shape[1] / pix_per_nm, img.shape[0] / pix_per_nm, 0]
    
    plt.figure(figsize = figuresize)
    fig = plt.imshow(img, cmap=AFM, extent=extent, vmin=vmin, vmax=vmax)

    plt.title(title)
    plt.xlabel('nm')
    plt.ylabel('nm')
    plt.colorbar(fig, label='Height (nm)')
    plt.show()
    
    return fig


def plot_clean(img, pix_per_nm, vmin=None, vmax=None, cmap_path='AFM_cmap.npy', figsize=(6, 6)):
   
    
    return show(img, pix_per_nm, 'Clean (target)', vmin, vmax, cmap_path, figsize)

def plot_noisy(img, pix_per_nm, noise = None, vmin=None, vmax=None, cmap_path='AFM_cmap.npy', figsize=(6, 6)):
    title = f'Noisy ({noise})'
    return show(img, pix_per_nm, title, vmin, vmax, cmap_path, figsize)
        


def save_tiff(out_path, img):
   
    out_path = Path(out_path)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    tifffile.imwrite(out_path, np.asarray(img, dtype=np.float32), imagej=True)
    return out_path

# protien_tr10_ca10_n10

def fmt(x):
    x = str(x)
    return x.replace('.','')
     

def clean_name(PDB_ID, tip_radius, cone_angle, orientation='fixed', renderer='exact'):
    """
    e.g. 6j6j_tr05_ca10_tilt30_exact
    Orientation and renderer are in the name so stacks made with different
    settings can never be mixed up or wrongly reused from the cache.
    """
    filename = f'{PDB_ID}_tr{fmt(tip_radius)}_ca{fmt(cone_angle)}_{orientation}_{renderer}'
    return filename


def noise_name(PDB_ID, noise, tip_radius, cone_angle, orientation='fixed', renderer='exact'):
    """e.g. 6j6j_tr05_ca10_tilt30_exact_n05"""
    filename = f'{clean_name(PDB_ID, tip_radius, cone_angle, orientation, renderer)}_n{fmt(noise)}'
    return filename


def name_seed(name):
    """
    Turn a filename string into a fixed integer seed.
    crc32 is a checksum: the same string always gives the same number, and
    different strings give (almost always) different numbers. So every
    protein / tip / cone / noise combination gets its own repeatable seed.
    """
    return zlib.crc32(name.encode())


def add_noise(clean, noise, seed):
    """
    Add Gaussian height noise (rms = `noise`, in nm) to a clean image or
    stack, using a fixed seed.

    Same clean + same noise + same seed  ->  exactly the same noisy stack,
    every time, on any computer. That is why the noisy tiffs no longer need
    to be saved: they can be rebuilt on demand from the clean tiff and the
    seed stored in data_table.csv (see export_noisy.py).

    This is the ONE place noise is made, used by both the dataset
    (unet/dataset.py) and the exporter, so they can never disagree.
    """
    rng = np.random.default_rng(seed)
    noisy = clean + rng.standard_normal(np.shape(clean)) * noise
    return noisy.astype(np.float32)




# =============================================================================
# # LOOPING THROUGH ITERATIONS OF SAME PROTIEN 
# here = Path(__file__).parent          # use Path.cwd() in a notebook
# 
# folder_path = here / PDB_ID
# 
# for file_path in folder_path.glob("*.tif"):
#     print(file_path)
# 
#     
# =============================================================================




# =============================================================================
# # create a figure and display the image with the afm colormap
# AFM = ListedColormap(np.load('AFM_cmap.npy'))
# # axes in nm: rows are x, columns are y
# extent = [0, img.shape[1] / pix_per_nm, img.shape[0] / pix_per_nm, 0]
# vmin, vmax = img_n.min(), img_n.max()  # same colour scale for both panels
# 
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# for ax, im, title in zip(axes, [img, img_n], ['Clean (target)', f'Noisy ({noise} nm rms)']):
#     h = ax.imshow(im, cmap=AFM, extent=extent, vmin=vmin, vmax=vmax)
#     ax.set_title(title)
#     ax.set_xlabel('nm')
#     ax.set_ylabel('nm')
# fig.colorbar(h, ax=axes, label='Height (nm)')
# plt.show()
# 
# # ground truth centre in image pixels (after rotation and z cut)
# cx = coords_s[:, 0].mean() - end_pos[0]
# cy = coords_s[:, 1].mean() - end_pos[2]
# 
# np.savez('sim_output.npz', clean=img, noisy=img_n, centre=[cx, cy], pix_per_nm=pix_per_nm)
# 
# end2= time()
# time_taken2 = end2 - start2
# print(f'Time to iterate through coords and calculate surface: {time_taken2}s')
# 
# =============================================================================

# =============================================================================
# # SHOWS PURE NOISE  
# resid = img_n - img
# print(f'residual mean {resid.mean():.4f} nm, std {resid.std():.4f} nm (set: {noise})')
# 
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# im = axes[0].imshow(resid.T, cmap='gray', extent=extent)
# axes[0].set_title('Noisy minus clean')
# axes[0].set_xlabel('x (nm)'); axes[0].set_ylabel('y (nm)')
# fig.colorbar(im, ax=axes[0], label='nm')
# axes[1].hist(resid.ravel(), bins=60)
# axes[1].set_xlabel('residual (nm)'); axes[1].set_title('Residual histogram')
# plt.show()
# =============================================================================
