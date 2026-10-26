#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet_parts.py

The three building blocks of the U-Net, following the structure of
uygarkurt/UNet-PyTorch (DoubleConv, DownSample, UpSample).

Nothing in here is specific to segmentation or denoising. These blocks just
turn an image into feature maps and back. What makes the network a denoiser
is the final layer and the loss function, which live in unet.py and
train_unet.py.

Shapes are written as (batch, channels, height, width), e.g. (B, 1, 64, 64)
for a batch of single-channel 64 x 64 AFM height images.

@author: lukethomas
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two 3x3 convolutions, each followed by a ReLU.
    These are the pairs of blue arrows in the original U-Net diagram.

    - kernel_size=3 means each output pixel looks at a 3x3 patch of the input.
    - padding=1 adds a one-pixel border so the image size is unchanged
      (64x64 in gives 64x64 out). The original 2015 U-Net paper had no padding,
      which shrinks the image; padding keeps the output the same size as the
      input, which we need since the output is a pixel-for-pixel clean image.
    - The first conv changes the number of channels (in -> out); the second
      keeps it the same (out -> out).
    - ReLU (max(0, x)) is the non-linearity. Without it, stacking convolutions
      would collapse into one big linear filter, i.e. no better than a
      hand-designed filter like a Gaussian blur.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # nn.Sequential runs these layers one after the other
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        return self.conv_op(x)


class DownSample(nn.Module):
    """
    One step down the left (encoder) side of the U.

    1. DoubleConv extracts features at the current resolution.
    2. MaxPool2d(2) halves the height and width by keeping the largest value
       in each 2x2 block (red arrows in the diagram).

    Returns BOTH:
    - `down`: the full-resolution features, kept for the skip connection.
      Skips carry fine spatial detail (sharp edges, small features) across to
      the decoder. For denoising this matters a lot: without them the network
      would have to rebuild fine detail from the coarse bottleneck and the
      output would come out blurred.
    - `p`: the pooled, half-size features, passed to the next level down.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)   # (B, out, H, W)
        p = self.pool(down)   # (B, out, H/2, W/2)
        return down, p


class UpSample(nn.Module):
    """
    One step up the right (decoder) side of the U.

    1. ConvTranspose2d (green arrow) doubles the height and width and halves
       the channels: (B, in, H, W) -> (B, in/2, 2H, 2W). It is a learnable
       upsampling, the network learns how best to fill in the new pixels.
    2. torch.cat joins this with the skip connection from the matching
       encoder level (grey arrow). The skip has in/2 channels too, so after
       concatenating we are back to `in` channels.
    3. DoubleConv mixes the coarse "what is here" information coming up from
       below with the fine "exactly where" information from the skip,
       giving `out` channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels // 2 so that, after concatenating with the skip
        # (which also has in_channels // 2), we get back to in_channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: features coming up from the level below (coarse)
        # x2: skip connection from the encoder at this level (fine)
        x1 = self.up(x1)                 # (B, in/2, 2H, 2W)
        x = torch.cat([x1, x2], dim=1)   # stack along channels -> (B, in, 2H, 2W)
        return self.conv(x)              # (B, out, 2H, 2W)
