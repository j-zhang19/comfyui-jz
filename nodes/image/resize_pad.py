"""jz Resize And Pad — letterbox an image to a target size, padding with any colour.

ComfyUI's own Resize And Pad Image does the same fit-and-centre, but its
padding_color is a two-option combo: white or black. This is that node with the
colour set free — type a hex value, or wire a solid-colour IMAGE into
color_image (jz Pad Calculator's fill_color output) and the colour is sampled
from it, the way jz Seam Repair already reads that output.

Also emits the padding region as a MASK (1 = bar, 0 = the fitted image), which
is what you need to inpaint the bars or composite something over them.

The fit is always scale-to-fit: min(tw/w, th/h), so a small image is scaled up
and a large one down, and the leftover is padded.
"""

import comfy.utils
import torch

from ...common.images import INTERPOLATION

# the two names the official node offers, so swapping this in keeps working
_NAMED = {"black": (0.0, 0.0, 0.0), "white": (1.0, 1.0, 1.0)}

_HEX = set("0123456789abcdef")


def _parse_color(text: str) -> tuple[float, float, float]:
    """'#rrggbb' / 'rrggbb' / '#rgb' / 'rgb' / 'black' / 'white' -> 0-1 floats."""
    s = (text or "").strip().lower()
    if s in _NAMED:
        return _NAMED[s]
    h = s[1:] if s.startswith("#") else s
    if len(h) == 3 and set(h) <= _HEX:
        h = "".join(c * 2 for c in h)
    if len(h) == 6 and set(h) <= _HEX:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    raise ValueError(
        f"jz Resize And Pad: padding_color {text!r} is not a colour — use "
        f"#rrggbb, #rgb, or one of {', '.join(_NAMED)}")


class jz_ResizeAndPad:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "resize_and_pad"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "target_height": ("INT", {"default": 1024, "min": 1, "max": 16384}),
                "padding_color": ("STRING", {
                    "default": "#000000",
                    "tooltip": "#rrggbb, #rgb, or black / white — ignored when "
                               "color_image is connected"}),
                "interpolation": (INTERPOLATION, {"default": "lanczos"}),
            },
            "optional": {
                "color_image": ("IMAGE", {
                    "tooltip": "a solid-colour image sampled at its top-left "
                               "pixel — jz Pad Calculator's fill_color output; "
                               "overrides padding_color when connected"}),
            },
        }

    def resize_and_pad(self, image, target_width, target_height, padding_color,
                       interpolation, color_image=None):
        b, h, w, c = image.shape
        if h == 0 or w == 0:
            raise ValueError(f"jz Resize And Pad: input image is empty ({w}x{h})")

        if color_image is not None:
            if color_image.shape[1] == 0 or color_image.shape[2] == 0:
                raise ValueError("jz Resize And Pad: color_image is empty (0 px) "
                                 "— it should be a solid-colour image")
            rgb = tuple(min(1.0, max(0.0, float(v)))
                        for v in color_image[0, 0, 0, :3])
        else:
            rgb = _parse_color(padding_color)

        # scale to fit, then centre. round (not truncate) fits a pixel better,
        # and the clamp keeps an extreme aspect ratio from collapsing an axis
        # to 0 px — a 4000x3 strip into 512x512 truncates to a height of 0.
        scale = min(target_width / w, target_height / h)
        new_w = max(1, min(target_width, round(w * scale)))
        new_h = max(1, min(target_height, round(h * scale)))

        chw = image.permute(0, 3, 1, 2)
        # already the right size: skip the resample entirely. lanczos round-trips
        # through PIL uint8 (comfy.utils.lanczos), so resampling a frame that
        # needs no resizing would quantize it for nothing.
        resized = chw if (new_w, new_h) == (w, h) else comfy.utils.common_upscale(
            chw, new_w, new_h, interpolation, "disabled")

        # canvas prefilled with the colour; a 4th channel is alpha, kept opaque
        fill = torch.tensor([rgb[ch] if ch < 3 else 1.0 for ch in range(c)],
                            dtype=image.dtype, device=image.device)
        out = fill.view(1, c, 1, 1).expand(
            b, c, target_height, target_width).clone()

        y = (target_height - new_h) // 2
        x = (target_width - new_w) // 2
        out[:, :, y:y + new_h, x:x + new_w] = resized

        mask = torch.ones((b, target_height, target_width),
                          dtype=image.dtype, device=image.device)
        mask[:, y:y + new_h, x:x + new_w] = 0.0

        return (out.permute(0, 2, 3, 1).contiguous(), mask)


NODE_CLASS_MAPPINGS = {"jz_ResizeAndPad": jz_ResizeAndPad}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ResizeAndPad": "jz Resize And Pad"}
