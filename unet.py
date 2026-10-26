#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unet.py

U-Net for denoising AFM height images, keeping the structure of
uygarkurt/UNet-PyTorch: 4 down steps, a bottleneck, 4 up steps with skip
connections, then a 1x1 output convolution.

What changed from the segmentation version, and why
---------------------------------------------------
1. `num_classes` -> `out_channels` (= 1).
   Segmentation outputs one score map per class and each pixel is assigned a
   label. Denoising outputs ONE continuous value per pixel: the predicted
   clean height. So there is a single output channel.

2. No sigmoid / softmax on the output.
   His version is trained with BCEWithLogitsLoss, which applies a sigmoid to
   squash outputs into 0-1 probabilities. Heights are not probabilities, so the
   output is left unbounded (a plain linear output), and training uses a
   regression loss (MSE) instead. See train_unet.py.

3. Optional residual learning (`residual=True`).
   Instead of predicting the clean image directly, the network predicts the
   NOISE and we subtract it: clean = noisy - noise_pred. The final output
   is then the input plus a learned correction. This is the trick used by
   DnCNN (Zhang et al. 2017) and usually trains faster and more stably for
   denoising because the network starts from "output = input" and only has
   to learn what to remove. Default is False to stay identical to his
   architecture; it is worth comparing both in the dissertation.

4. `base_channels` argument.
   His network is fixed at 64 -> 128 -> 256 -> 512 -> 1024 channels
   (about 31 million parameters). With only a few proteins that is a lot of
   capacity for not much data. base_channels=64 reproduces his network
   exactly; 32 or 16 gives the same shape with 4x or 16x fewer parameters.

Input size: the image is halved 4 times, so H and W must be divisible by
2**4 = 16. Your 64x64 SimAFM canvases go 64 -> 32 -> 16 -> 8 -> 4.

@author: lukethomas
"""

import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 residual=False):
        super().__init__()
        c = base_channels      # 64 in the original
        self.residual = residual

        # ---- Encoder (left side of the U) ----
        # Each step: double the channels, halve the image size.
        # Early levels see small neighbourhoods (fine detail, noise texture),
        # deeper levels see large areas (overall molecule shape).
        self.down_convolution_1 = DownSample(in_channels, c)     # 64x64 -> 32x32
        self.down_convolution_2 = DownSample(c, c * 2)           # 32x32 -> 16x16
        self.down_convolution_3 = DownSample(c * 2, c * 4)       # 16x16 -> 8x8
        self.down_convolution_4 = DownSample(c * 4, c * 8)       # 8x8   -> 4x4

        # ---- Bottleneck (bottom of the U) ----
        # Lowest resolution, most channels. Each pixel here "sees" a large
        # part of the image, so this is where the network uses global
        # context, e.g. telling a real molecular feature from a noise spike.
        self.bottle_neck = DoubleConv(c * 8, c * 16)              # 4x4

        # ---- Decoder (right side of the U) ----
        # Each step: upsample x2, join with the matching skip, halve channels.
        self.up_convolution_1 = UpSample(c * 16, c * 8)           # 4x4   -> 8x8
        self.up_convolution_2 = UpSample(c * 8, c * 4)            # 8x8   -> 16x16
        self.up_convolution_3 = UpSample(c * 4, c * 2)            # 16x16 -> 32x32
        self.up_convolution_4 = UpSample(c * 2, c)                # 32x32 -> 64x64

        # ---- Output layer ----
        # A 1x1 convolution is a per-pixel weighted sum of the c feature
        # channels, turning them into a single height value per pixel.
        # (In his segmentation version this produced num_classes score maps.)
        self.out = nn.Conv2d(in_channels=c, out_channels=out_channels,
                             kernel_size=1)

    def forward(self, x):
        # x: noisy image, shape (B, 1, 64, 64)

        # Encoder: keep down_i for the skip connections, pass p_i further down
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        # Decoder: each step is fed the matching encoder output (skip)
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)          # (B, 1, 64, 64)

        if self.residual:
            # Network output is interpreted as the predicted noise,
            # so the clean estimate is the input with that noise removed.
            return x - out
        # Otherwise the network output IS the clean-image estimate.
        return out


if __name__ == "__main__":
    # Quick sanity check: run `python unet.py`
    # A fake batch of 2 noisy 64x64 images goes in, the same shape comes out.
    model = UNet(in_channels=1, out_channels=1)
    x = torch.randn(2, 1, 64, 64)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"in {tuple(x.shape)} -> out {tuple(y.shape)}, "
          f"{n_params / 1e6:.1f} M parameters")
