"""jz OpenAI Image : gpt-image generation and editing on the OpenAI API.

No image wired: POST /images/generations. Image(s) wired: POST /images/edits
in its JSON form (`images` as data URLs, up to 16), so no multipart. A MASK
applies to the first image: white (1) = the area to change.

Synchronous; complex prompts can take ~2 min. Billed per token, reported in
`usage` and turned into `cost`. Only what you set is sent: `auto` omits the
field, because support varies per model (xhigh/max exist only on 2.5,
input_fidelity is rejected by gpt-image-2).
"""
import base64
import io

import numpy as np
import torch
from PIL import Image

from ...common.images import pil_to_b64, pils_to_batch
from ...common.nodes import scalar
from ...common.openai import (CURATED_IMAGE, api_post, dumps, image_cost,
                              image_refs, model_ids, resolve_model)
from ...common.secrets import openai_key

SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536", "2048x2048",
         "2048x1152", "1152x2048", "3840x2160", "2160x3840", "custom"]
QUALITIES = ["auto", "low", "medium", "high", "xhigh", "max"]
MAX_REFS = 16
_PX_MIN, _PX_MAX, _EDGE_MAX = 655_360, 8_294_400, 3840


def _custom_size(w: int, h: int) -> str:
    """The documented constraints for arbitrary sizes, checked before spending."""
    errs = []
    if w % 16 or h % 16:
        errs.append("both sides must be multiples of 16")
    if max(w, h) > _EDGE_MAX:
        errs.append(f"max edge is {_EDGE_MAX}")
    if max(w, h) > 3 * min(w, h):
        errs.append("aspect ratio must be within 1:3 to 3:1")
    if not _PX_MIN <= w * h <= _PX_MAX:
        errs.append(f"total pixels must be {_PX_MIN:,} to {_PX_MAX:,}")
    if errs:
        raise ValueError(f"jz OpenAI Image: {w}x{h} invalid: {'; '.join(errs)}. "
                         f"custom sizes need gpt-image-2 or newer")
    return f"{w}x{h}"


def _mask_ref(mask, size) -> str:
    """ComfyUI MASK (1 = change) -> PNG whose alpha is 0 where the edit goes,
    resized to the first image, which is what the API requires."""
    m = (mask if isinstance(mask, list) else [mask])[0]
    m = m[0] if m.dim() == 3 else m
    a = ((1.0 - m.clamp(0, 1)).cpu().numpy() * 255).astype(np.uint8)
    alpha = Image.fromarray(a, "L").resize(size, Image.NEAREST)
    rgba = Image.new("RGBA", size, (0, 0, 0, 255))
    rgba.putalpha(alpha)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _first_size(image):
    t = (image if isinstance(image, list) else [image])[0]
    return (int(t.shape[2]), int(t.shape[1]))


def _decode(items):
    """data[].b64_json -> (RGB batch, MASK from alpha: 1 = transparent)."""
    pils = [Image.open(io.BytesIO(base64.b64decode(it["b64_json"])))
            for it in items if it.get("b64_json")]
    if not pils:
        raise RuntimeError(f"jz OpenAI Image: no b64_json in "
                           f"{[sorted(i) for i in items]}")
    batch = pils_to_batch(pils)
    h, w = batch.shape[1:3]
    masks = []
    for p in pils:
        p = p if p.size == (w, h) else p.resize((w, h), Image.LANCZOS)
        if "A" in p.getbands():
            a = np.array(p.getchannel("A")).astype(np.float32) / 255.0
            masks.append(torch.from_numpy(1.0 - a))
        else:
            masks.append(torch.zeros((h, w)))
    return batch, torch.stack(masks)


class jz_OpenAIImage:
    CATEGORY = "jz/api"
    # a LIST of references must not fan out into one billed call per item
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("images", "mask", "cost", "usage")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a cute dog"}),
                "model": (model_ids("image", "gpt-image", CURATED_IMAGE),
                          {"default": CURATED_IMAGE[0]}),
                "size": (SIZES, {
                    "default": "1024x1024",
                    "tooltip": "auto lets the model pick. anything beyond the "
                               "first three, and custom, needs gpt-image-2+"}),
                "custom_width": ("INT", {"default": 2048, "min": 16,
                                         "max": 3840, "step": 16}),
                "custom_height": ("INT", {"default": 2048, "min": 16,
                                          "max": 3840, "step": 16}),
                "quality": (QUALITIES, {
                    "default": "auto",
                    "tooltip": "auto omits it. xhigh/max: gpt-image-2.5 only. "
                               "cost scales steeply with quality"}),
                "background": (["auto", "opaque", "transparent"], {
                    "default": "auto",
                    "tooltip": "transparent needs png or webp; the alpha comes "
                               "out on the mask output"}),
                "output_format": (["png", "jpeg", "webp"], {"default": "png"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10,
                              "tooltip": "each image is billed"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "wired = edit. every frame is one "
                                               f"reference, max {MAX_REFS}"}),
                "mask": ("MASK", {"tooltip": "applies to the first image; "
                                             "1 = area to change"}),
                "input_fidelity": (["auto", "low", "high"], {
                    "default": "auto",
                    "tooltip": "edits only. auto omits it, which gpt-image-2+ "
                               "requires"}),
                "moderation": (["auto", "low"], {"default": "auto"}),
                "output_compression": ("INT", {
                    "default": 100, "min": 0, "max": 100,
                    "tooltip": "jpeg/webp only"}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            },
        }

    def generate(self, prompt, model, size, custom_width, custom_height,
                 quality, background, output_format, n, image=None, mask=None,
                 input_fidelity="auto", moderation="auto",
                 output_compression=100, custom_model="", api_key=""):
        # INPUT_IS_LIST: every widget arrives as a 1-element list
        prompt = str(scalar(prompt, ""))
        size, quality = str(scalar(size)), str(scalar(quality))
        background = str(scalar(background))
        output_format = str(scalar(output_format))
        n = int(scalar(n, 1))
        input_fidelity = str(scalar(input_fidelity, "auto"))
        moderation = str(scalar(moderation, "auto"))
        output_compression = int(scalar(output_compression, 100))
        api_key = str(scalar(api_key, ""))
        if not prompt.strip():
            raise ValueError("jz OpenAI Image: prompt is empty")
        model = resolve_model(str(scalar(model)), str(scalar(custom_model, "")))
        if background == "transparent" and output_format == "jpeg":
            raise ValueError("jz OpenAI Image: transparent needs png or webp")

        payload = {"model": model, "prompt": prompt, "n": n,
                   "output_format": output_format}
        if size == "custom":
            payload["size"] = _custom_size(int(scalar(custom_width)),
                                           int(scalar(custom_height)))
        elif size != "auto":
            payload["size"] = size
        for k, v in (("quality", quality), ("background", background),
                     ("moderation", moderation)):
            if v != "auto":
                payload[k] = v
        if output_format != "png" and output_compression != 100:
            payload["output_compression"] = output_compression

        refs = image_refs(image) if image is not None else []
        if len(refs) > MAX_REFS:
            raise ValueError(f"jz OpenAI Image: {len(refs)} reference images, "
                             f"the api takes {MAX_REFS}")
        if refs:
            path = "/images/edits"
            payload["images"] = [{"image_url": u} for u in refs]
            if mask is not None and mask[0] is not None:
                payload["mask"] = {"image_url": _mask_ref(mask, _first_size(image))}
            if input_fidelity != "auto":
                payload["input_fidelity"] = input_fidelity
        else:
            path = "/images/generations"
            if mask is not None and mask[0] is not None:
                raise ValueError("jz OpenAI Image: a mask needs an image to edit")

        print(f"[jz openai-image] {path} {model} size={payload.get('size', 'auto')} "
              f"quality={quality} n={n} refs={len(refs)}", flush=True)
        data = api_post(path, payload, openai_key(api_key), timeout=600)

        batch, alpha = _decode(data.get("data") or [])
        usage = data.get("usage") or {}
        cost = image_cost(model, usage)
        print(f"[jz openai-image] {batch.shape[0]} image(s) "
              f"{batch.shape[2]}x{batch.shape[1]} {cost}", flush=True)
        return (batch, alpha, cost, dumps(usage))


NODE_CLASS_MAPPINGS = {"jz_OpenAIImage": jz_OpenAIImage}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_OpenAIImage": "jz OpenAI Image"}
