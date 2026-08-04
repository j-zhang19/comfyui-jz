"""jz Resize Long Edge — normalize a list (or batch) of images to one size.

Accepts a ComfyUI image LIST (mixed sizes welcome) or a regular batch, and
resizes every frame so its longest side equals `long_edge`. Outputs a LIST
(one frame per item) so mixed aspect ratios survive — batches can't hold
mixed sizes, lists can.
"""
from ...common.images import pil_to_tensor_batch, tensor_frame_to_pil

from PIL import Image


def _scalar(v, default=None):
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default


class jz_ResizeLongEdge:
    CATEGORY = "jz/util"
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
        }

    def resize(self, image, long_edge, upscale):
        target = int(_scalar(long_edge, 1024))
        up = bool(_scalar(upscale, True))
        tensors = image if isinstance(image, list) else [image]
        out = []
        for t in tensors:
            if t is None:
                continue
            for frame in t:  # unroll batches into list items
                pil = tensor_frame_to_pil(frame)
                m = max(pil.size)
                if m != target and (up or m > target):
                    s = target / m
                    pil = pil.resize((max(1, round(pil.width * s)),
                                      max(1, round(pil.height * s))),
                                     Image.LANCZOS)
                out.append(pil_to_tensor_batch(pil))
        if not out:
            raise ValueError("jz Resize Long Edge: no images provided")
        return (out, [len(out)])


NODE_CLASS_MAPPINGS = {"jz_ResizeLongEdge": jz_ResizeLongEdge}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ResizeLongEdge": "jz Resize Long Edge (list)"}
