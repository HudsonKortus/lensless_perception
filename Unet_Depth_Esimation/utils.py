import os, math, argparse, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
import time

import torch, torch.nn as nn




def to_tensor_img(pil_img):
    arr = np.array(pil_img, dtype=np.uint8)
    return torch.from_numpy(arr).permute(2,0,1).float() / 255.0

def save_depth_png(depth_m: np.ndarray,
                   out_png: str,
                   mode: str = "fixed",
                   vmin: float = 0.3,
                   vmax: float = 10.0,
                   lo_p: float = 1.0,
                   hi_p: float = 99.0):
    """
    Save a depth map as an 8-bit PNG for visualization.

    mode = "fixed": clamp to [vmin, vmax] meters, then normalize.
    mode = "auto" : compute percentiles [lo_p, hi_p] per-image, then normalize.
    """
    d = depth_m.astype(np.float32).copy()
    if mode == "fixed":
        d = np.clip(d, vmin, vmax)
        d = (d - vmin) / (vmax - vmin + 1e-8)
    elif mode == "auto":
        lo, hi = np.percentile(d, [lo_p, hi_p])
        if hi <= lo:  # fallback if degenerate
            d[:] = 0.0
        else:
            d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown viz mode: {mode}")
    d8 = (d * 255.0).astype(np.uint8)
    imageio.imwrite(out_png, d8)

def read_depth_png_auto(p: Path):
    im = Image.open(p)
    arr = np.array(im)
    arr = arr.astype(np.float32)
    if arr.dtype == np.uint16 or arr.max() > 50.0:
        arr = arr / 1000.0  # mm -> m
    return arr

def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)

def CombineImages(pred, label, rgb):
    pred = pred.detach().cpu().numpy().squeeze()
    label = label.detach().cpu().numpy().squeeze()
    rgb = rgb.detach().cpu().numpy()
    
    gray_array = 0.299 * rgb[0, :, :] + 0.587 * rgb[1, :, :] + 0.114 * rgb[2, :, :]

    # Concatenate images horizontally
    combined_image_np = np.concatenate((pred, label, gray_array), axis=1)
    combined_image_np = (np.clip(combined_image_np, 0, 1)*255).astype(np.uint8)
    return combined_image_np