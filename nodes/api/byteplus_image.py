"""jz BytePlus Seedream — image generation and editing on BytePlus ModelArk.

POST /images/generations, synchronous (~8s). Wire an IMAGE in and every frame
becomes a reference, which is how editing / multi-reference blending works.

Parameter support varies per model and IS enforced: seedream-4-0 rejects
`output_format` outright ("not supported by the current model"), and 4k is
rejected by dola-seedream-5-0-pro. So only what you actually set is sent.

Billing is per output token (a 1k image is ~4096), reported exactly in `usage`.
"""
import base64
import io
import json

from PIL import Image

from ...common.byteplus import (CURATED_IMAGE, IMAGE_SIZES, REGIONS, api_post, image_refs,
                                model_ids, resolve_model)
from ...common.secrets import byteplus_key
from ...common.http import SESSION
from ...common.images import pils_to_batch

_PX_MIN, _PX_MAX = 921_600, 16_777_216


class jz_BytePlusSeedream:
    CATEGORY = "jz/api"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls", "usage")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a cute dog"}),
                "model": (model_ids("image", CURATED_IMAGE),
                          {"default": CURATED_IMAGE[0]}),
                "size": (IMAGE_SIZES, {
                    "default": "2k",
                    "tooltip": "1k/2k/4k, or custom WxH. not every model takes "
                               "4k — dola-seedream-5-0-pro rejects it"}),
                "custom_width": ("INT", {"default": 1024, "min": 256,
                                         "max": 4096, "step": 8}),
                "custom_height": ("INT", {"default": 1024, "min": 256,
                                          "max": 4096, "step": 8}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8,
                              "tooltip": "each image is billed separately"}),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "the api default is ON, so this is sent "
                               "explicitly"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1,
                                 "tooltip": "-1 omits it. left on 'fixed' so a "
                                            "re-queue cannot silently spend"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "reference images for editing — "
                                               "every frame becomes one ref"}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "region": (list(REGIONS), {"default": "ap-southeast"}),
            },
        }

    def generate(self, prompt, model, size, custom_width, custom_height, n,
                 watermark, seed, image=None, custom_model="", api_key="",
                 region="ap-southeast"):
        if not prompt.strip():
            raise ValueError("jz BytePlus Seedream: prompt is empty")
        model = resolve_model(model, custom_model)

        if size == "custom":
            w, h = int(custom_width), int(custom_height)
            px = w * h
            if not _PX_MIN <= px <= _PX_MAX:
                raise ValueError(
                    f"jz BytePlus Seedream: {w}x{h} is {px:,} pixels — the api "
                    f"accepts {_PX_MIN:,} to {_PX_MAX:,}")
            size_arg = f"{w}x{h}"
        else:
            size_arg = size

        payload = {"model": model, "prompt": prompt, "size": size_arg,
                   "n": int(n), "watermark": bool(watermark),
                   "response_format": "url"}
        if int(seed) >= 0:
            payload["seed"] = int(seed)
        refs = image_refs(image)
        if refs:
            payload["image"] = refs[0] if len(refs) == 1 else refs

        print(f"[jz seedream] {model} size={size_arg} n={n} refs={len(refs)}",
              flush=True)
        data = api_post(region, "/images/generations", payload,
                        byteplus_key(api_key), timeout=600)

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"jz BytePlus Seedream: no images returned: "
                               f"{json.dumps(data)[:300]}")

        pils, urls = [], []
        for it in items:
            url = it.get("url")
            if url:
                urls.append(url)
                pils.append(Image.open(io.BytesIO(
                    SESSION.get(url, timeout=300).content)).convert("RGB"))
            elif it.get("b64_json"):
                urls.append("<b64_json>")
                pils.append(Image.open(io.BytesIO(
                    base64.b64decode(it["b64_json"]))).convert("RGB"))
        if not pils:
            raise RuntimeError(f"jz BytePlus Seedream: no image bytes in "
                               f"{[sorted(i) for i in items]}")

        batch = pils_to_batch(pils)
        usage = data.get("usage") or {}
        print(f"[jz seedream] {len(pils)} image(s) {pils[0].size[0]}x"
              f"{pils[0].size[1]} tokens={usage.get('total_tokens', '?')}",
              flush=True)
        return (batch, "\n".join(urls), json.dumps(usage, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {"jz_BytePlusSeedream": jz_BytePlusSeedream}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_BytePlusSeedream": "jz BytePlus Seedream (image)"}
