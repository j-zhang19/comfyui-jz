"""jz Resize Long Edge — normalize a list (or batch) of images to one size.

Accepts a ComfyUI image LIST (mixed sizes welcome) or a regular batch, and
resizes every frame so its longest side equals `long_edge`. Outputs a LIST
(one frame per item) so mixed aspect ratios survive — batches can't hold
mixed sizes, lists can.

Resampling goes through comfy.utils.common_upscale, the same call jz Resize
And Pad uses, so `interpolation` offers the same five methods and a frame that
is already at the target size is passed through untouched.
"""
import comfy.utils

from ...common.images import INTERPOLATION
from ...common.nodes import scalar


class jz_ResizeLongEdge:
    CATEGORY = "jz/image"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "count")
    FUNCTION = "resize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "long_edge": ("INT", {"default": 1024, "min": 16, "max": 8192}),
                "upscale": ("BOOLEAN", {"default": True,
                                        "tooltip": "off = only shrink larger "
                                                   "images, keep smaller ones"}),
            },
            # appended (append-only rule). was hardcoded to lanczos before
            "optional": {
                "interpolation": (INTERPOLATION, {
                    "default": "lanczos",
                    "tooltip": "lanczos is the sharpest but round-trips through "
                               "8-bit; area/bicubic/bilinear stay in float"}),
            },
        }

    def resize(self, image, long_edge, upscale, interpolation="lanczos"):
        target = int(scalar(long_edge, 1024))
        up = bool(scalar(upscale, True))
        interp = str(scalar(interpolation, "lanczos"))
        tensors = image if isinstance(image, list) else [image]
        out = []
        for t in tensors:
            if t is None:
                continue
            for frame in t:  # unroll batches into list items
                h, w = int(frame.shape[0]), int(frame.shape[1])
                m = max(w, h)
                if not m or m == target or not (up or m > target):
                    out.append(frame.unsqueeze(0))  # nothing to do, no resample
                    continue
                s = target / m
                chw = frame.unsqueeze(0).permute(0, 3, 1, 2)
                chw = comfy.utils.common_upscale(
                    chw, max(1, round(w * s)), max(1, round(h * s)),
                    interp, "disabled")
                out.append(chw.permute(0, 2, 3, 1).contiguous())
        if not out:
            raise ValueError("jz Resize Long Edge: no images provided")
        return (out, [len(out)])


NODE_CLASS_MAPPINGS = {"jz_ResizeLongEdge": jz_ResizeLongEdge}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ResizeLongEdge": "jz Resize Long Edge (list)"}
