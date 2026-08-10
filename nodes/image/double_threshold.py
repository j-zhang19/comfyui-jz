"""jz Double Threshold — two-sided binarization with a transparent middle.

Luma >= high goes pure white, luma <= low goes pure black, everything in
between is transparent (alpha 0). The classic trimap construction: the
transparent band is the "unknown" zone for matting, or simply drops the
midtones for compositing (two-sided luma key).

Outputs:
- image: RGBA — white/black opaque, midtones transparent (jz Composite
  blends 4-channel sources through their alpha automatically)
- trimap: MASK with 1 = white, 0 = black, 0.5 = unknown band
- alpha:  MASK with 1 = decided (kept), 0 = transparent band
"""
import torch

# BT.601 luma, same weighting as the seam carve node
_LUMA = (0.299, 0.587, 0.114)


class jz_DoubleThreshold:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "trimap", "alpha")
    FUNCTION = "threshold"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0,
                                  "step": 0.01,
                                  "tooltip": "luma at or below this -> black"}),
                "high": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0,
                                   "step": 0.01,
                                   "tooltip": "luma at or above this -> white"}),
            },
        }

    def threshold(self, image, low, high):
        if low >= high:
            raise ValueError(f"jz Double Threshold: low ({low}) must be "
                             f"below high ({high})")

        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        luma = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b

        white = luma >= high
        black = luma <= low
        alpha = (white | black).to(image.dtype)

        rgb = white.to(image.dtype).unsqueeze(-1).expand(-1, -1, -1, 3)
        rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1).contiguous()

        trimap = torch.full_like(luma, 0.5)
        trimap[white] = 1.0
        trimap[black] = 0.0

        return (rgba, trimap, alpha)


NODE_CLASS_MAPPINGS = {"jz_DoubleThreshold": jz_DoubleThreshold}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_DoubleThreshold": "jz Double Threshold"}
