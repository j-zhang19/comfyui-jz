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
            # appended (append-only rule). LoadImage strips a PNG's alpha into
            # its MASK output — wire it here so transparent input pixels stay
            # transparent instead of being thresholded on their (black) RGB
            "optional": {
                "input_alpha": ("MASK", {"tooltip": "input transparency; "
                                                    "LoadImage's MASK output "
                                                    "goes here"}),
                "invert_input_alpha": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "LoadImage masks are 1 = transparent; keep on "
                               "when wiring that, turn off if your mask is "
                               "1 = opaque"}),
            },
        }

    def threshold(self, image, low, high, input_alpha=None,
                  invert_input_alpha=True):
        if low >= high:
            raise ValueError(f"jz Double Threshold: low ({low}) must be "
                             f"below high ({high})")

        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        luma = _LUMA[0] * r + _LUMA[1] * g + _LUMA[2] * b

        # input opacity: 4th image channel and/or the input_alpha socket
        opacity = torch.ones_like(luma)
        if image.shape[-1] == 4:
            opacity = opacity * image[..., 3]
        if input_alpha is not None:
            if input_alpha.dim() == 2:
                input_alpha = input_alpha.unsqueeze(0)
            # LoadImage emits a 64x64 constant placeholder mask when the
            # image has no alpha channel — treat it as "no mask wired"
            if (input_alpha.shape[-2:] == (64, 64)
                    and luma.shape[-2:] != (64, 64)
                    and (input_alpha == input_alpha.flatten()[0]).all()):
                input_alpha = None
            elif input_alpha.shape[-2:] != luma.shape[-2:]:
                raise ValueError(
                    f"jz Double Threshold: input_alpha "
                    f"{input_alpha.shape[-1]}x{input_alpha.shape[-2]} does not "
                    f"match image {luma.shape[-1]}x{luma.shape[-2]}")
        if input_alpha is not None:
            if invert_input_alpha:
                input_alpha = 1.0 - input_alpha
            opacity = opacity * input_alpha

        white = luma >= high
        black = luma <= low
        alpha = (white | black).to(image.dtype) * opacity

        rgb = white.to(image.dtype).unsqueeze(-1).expand(-1, -1, -1, 3)
        rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1).contiguous()

        trimap = torch.full_like(luma, 0.5)
        trimap[white] = 1.0
        trimap[black] = 0.0
        trimap[opacity < 0.5] = 0.5  # input-transparent pixels stay undecided

        return (rgba, trimap, alpha)


NODE_CLASS_MAPPINGS = {"jz_DoubleThreshold": jz_DoubleThreshold}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_DoubleThreshold": "jz Double Threshold"}
