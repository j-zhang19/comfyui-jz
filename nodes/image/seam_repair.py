"""jz Seam Repair — clean a thin leftover fill-color seam at the canvas edge.

Mirrors the production RunPod worker (vermeer-runpod-outpainting), specifically
``src/metrics.py::repair_fill_residue``. Pure numpy/scipy, so it carries no
dependency beyond what the pad node already needs.
"""

import numpy as np
import torch
from PIL import Image
from scipy import ndimage



def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE tensor (B,H,W,C float 0-1) -> [H,W,C] uint8."""
    return (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)


def _fill_rgb_from_image(fill_color: torch.Tensor) -> tuple[int, int, int]:
    """Sample the solid fill-color image (from jz Pad Calculator) as RGB."""
    px = (fill_color[0, 0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return int(px[0]), int(px[1]), int(px[2])


class jz_SeamRepair:
    """Clean a thin leftover fill-color seam at the outpaint canvas edge.

    Mirrors ``metrics.repair_fill_residue``: only acts when the *full-region*
    residue at ``tol`` is small (<= ``max_fraction``); a larger residue means the
    model echoed the whole pad instead of outpainting (a real failure) and is left
    untouched so it stays visible rather than being smeared over.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "outpaint_mask": ("MASK",),
                "fill_color": ("IMAGE",),
                "tol": ("INT", {"default": 80, "min": 0, "max": 255}),
                "max_fraction": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "repair_tol": ("INT", {"default": 120, "min": 0, "max": 255}),
                "edge_band": ("INT", {"default": 5, "min": 1, "max": 128}),
                "dilate": ("INT", {"default": 3, "min": 0, "max": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "residue")
    FUNCTION = "repair"
    CATEGORY = "jz/image"

    def repair(
        self,
        generated_image: torch.Tensor,
        outpaint_mask: torch.Tensor,
        fill_color: torch.Tensor,
        tol: int,
        max_fraction: float,
        repair_tol: int,
        edge_band: int,
        dilate: int,
    ):
        arr = _tensor_to_uint8(generated_image)  # H,W,3 uint8
        h, w = arr.shape[:2]

        # Mask -> boolean region at the image resolution (1.0 = area to fill).
        mask_np = outpaint_mask[0].cpu().numpy()
        if mask_np.shape != (h, w):
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
                (w, h), Image.NEAREST
            )
            mask_np = np.asarray(mask_pil).astype(np.float32) / 255.0
        region = mask_np > 0.5
        n = int(region.sum())
        if n == 0:
            return (generated_image, 0.0)

        fill = np.array(_fill_rgb_from_image(fill_color), dtype=np.int16)
        diff = np.abs(arr.astype(np.int16) - fill).max(axis=2)

        # Full-region residue at the QC tolerance. A large value is a real echo of
        # the pad — leave it untouched so it stays visible (repairing would smear).
        frac = float(int(((diff <= tol) & region).sum()) / n)
        if frac == 0.0 or frac > max_fraction:
            return (generated_image, frac)

        # Only the outermost edge_band px of the canvas, within the outpaint
        # region: restricting the search lets us use a generous colour tolerance
        # without ever matching interior scene content.
        b = int(edge_band)
        edge = np.zeros((h, w), dtype=bool)
        edge[:b, :] = edge[-b:, :] = edge[:, :b] = edge[:, -b:] = True
        bad = (diff <= repair_tol) & region & edge
        if not bad.any():
            return (generated_image, frac)

        if dilate > 0:
            bad = ndimage.binary_dilation(bad, iterations=int(dilate)) & region

        # Map every bad pixel to its nearest good pixel and copy that colour in.
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        fixed = arr[tuple(idx)]

        out = torch.from_numpy(fixed.astype(np.float32) / 255.0).unsqueeze(0)
        return (out, frac)


# key kept as "GeminiFillSeamRepair" so saved workflows still resolve
NODE_CLASS_MAPPINGS = {"GeminiFillSeamRepair": jz_SeamRepair}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiFillSeamRepair": "jz Seam Repair"}
