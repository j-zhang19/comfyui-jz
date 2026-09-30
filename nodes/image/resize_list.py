"""jz Resize Long Edge — normalize a list (or batch) of images to one size.

Accepts a ComfyUI image LIST (mixed sizes welcome) or a regular batch, and
resizes every frame so its longest side equals `long_edge`. Outputs a LIST
(one frame per item) so mixed aspect ratios survive — batches can't hold
mixed sizes, lists can.

Resampling goes through comfy.utils.common_upscale, the same call jz Resize
And Pad uses, so `interpolation` offers the same five methods and a frame that
is already at the target size is passed through untouched.

The appended `batch` output is the same frames as one tensor, for the nodes
that need a real batch rather than a list. A tensor cannot hold mixed sizes,
so each frame is centred on a canvas big enough for all of them — when every
frame already matches (a uniform set of inputs) nothing is padded and the
batch is just a stack.
"""
import comfy.utils
import torch

from ...common.images import INTERPOLATION
from ...common.nodes import scalar


def stack_padded(tensors: list) -> torch.Tensor:
    """One tensor from frames of differing size, each centred, padded black.

    Mixed channel counts (an RGBA beside an RGB) are levelled up with an
    opaque alpha, and the padding itself is opaque — matching jz Resize And Pad.
    """
    mh = max(t.shape[1] for t in tensors)
    mw = max(t.shape[2] for t in tensors)
    mc = max(t.shape[3] for t in tensors)
    frames = []
    for t in tensors:
        f = t[0]
        h, w, c = f.shape
        if c < mc:
            f = torch.cat(
                [f, torch.ones(h, w, mc - c, dtype=f.dtype, device=f.device)], -1)
        if (h, w) != (mh, mw):
            canvas = torch.zeros(mh, mw, mc, dtype=f.dtype, device=f.device)
            if mc == 4:
                canvas[..., 3] = 1.0
            y, x = (mh - h) // 2, (mw - w) // 2
            canvas[y:y + h, x:x + w] = f
            f = canvas
        frames.append(f)
    return torch.stack(frames, dim=0)


class jz_ResizeLongEdge:
    CATEGORY = "jz/image"
    INPUT_IS_LIST = True
    # `batch` appended (append-only rule) and marked scalar: a False slot is
    # taken as one value, a True slot is extend()ed
    OUTPUT_IS_LIST = (True, True, False)
    RETURN_TYPES = ("IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("images", "count", "batch")
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
        return (out, [len(out)], stack_padded(out))


NODE_CLASS_MAPPINGS = {"jz_ResizeLongEdge": jz_ResizeLongEdge}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ResizeLongEdge": "jz Resize Long Edge (list)"}
