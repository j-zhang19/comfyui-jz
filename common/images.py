"""Shared tensor/PIL/base64 image helpers for ComfyUI nodes."""
import base64
import io

import numpy as np
import torch
from PIL import Image


def tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    arr = (frame.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def resize_long_edge(pil: Image.Image, px: int) -> Image.Image:
    if max(pil.size) <= px:
        return pil
    scale = px / max(pil.size)
    return pil.resize((max(1, round(pil.width * scale)),
                       max(1, round(pil.height * scale))), Image.LANCZOS)


def pil_to_data_url(pil: Image.Image) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def batch_to_data_urls(image: torch.Tensor, max_edge: int) -> list[str]:
    """Every frame of a BHWC batch becomes one data-URL image."""
    return [pil_to_data_url(resize_long_edge(tensor_frame_to_pil(f), max_edge))
            for f in image]


def pil_to_tensor_batch(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# Resample methods comfy.utils.common_upscale accepts, lanczos first because it
# is this pack's quality default. Note lanczos round-trips through PIL uint8
# (comfy.utils.lanczos); the other four stay in float via torch.interpolate.
INTERPOLATION = ["lanczos", "area", "bicubic", "nearest-exact", "bilinear"]


# BT.601 luma. The arithmetic is identical for numpy and torch arrays, so the
# seam-carve (numpy) and threshold/sanity (torch) nodes share one definition.
LUMA = (0.299, 0.587, 0.114)


def luma(rgb):
    """Perceptual brightness of a ...x3 RGB array (numpy or torch)."""
    return LUMA[0] * rgb[..., 0] + LUMA[1] * rgb[..., 1] + LUMA[2] * rgb[..., 2]
