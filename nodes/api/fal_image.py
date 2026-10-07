"""jz fal Image — image generation and editing on fal.ai.

One node for every image model on fal (~1500 across the catalogue), because
the model-specific arguments go in a `params` JSON widget that is validated
against that model's published OpenAPI schema before anything is submitted.
A typo'd key, a bad enum or a missing required field fails locally and free.

Wire an IMAGE in and the schema decides how it is sent: models declaring
`image_urls` get every frame, models declaring `image_url` get the first.
"""
import json

from ...common.fal import CURATED_IMAGE, model_ids
from ...common.http import SESSION
from ...common.images import pils_to_batch
from ...common.nodes import scalar
from ._fal_common import run


class jz_FalImage:
    CATEGORY = "jz/api"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "urls", "info")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_ids("image", CURATED_IMAGE),
                          {"default": CURATED_IMAGE[0]}),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a cute dog"}),
                "params": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "model-specific arguments as json, e.g. "
                               '{"image_size": "landscape_16_9", '
                               '"num_images": 2}. checked against the model\'s '
                               "schema before anything is billed; the console "
                               "prints what it accepts"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "reference image(s) for editing"}),
                "custom_model": ("STRING", {
                    "default": "",
                    "tooltip": "any fal endpoint id, used when model is custom"}),
                "api_key": ("STRING", {"default": ""}),
                "poll_timeout": ("INT", {"default": 900, "min": 30, "max": 7200}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
            },
        }

    def generate(self, model, prompt, params, image=None, custom_model="",
                 api_key="", poll_timeout=900, poll_interval=3):
        model = str(scalar(model, ""))
        if model == "custom":
            custom = str(scalar(custom_model, "")).strip()
            if not custom:
                raise RuntimeError("jz fal Image: model is 'custom' but "
                                   "custom_model is empty")
            model = custom
        _, out = run(model, prompt, params, image, api_key,
                     poll_timeout, poll_interval, tag="jz fal image")

        items = out.get("images") or []
        urls = [i.get("url") for i in items if i.get("url")]
        if not urls:
            raise RuntimeError(f"jz fal Image: no images in the result: "
                               f"{json.dumps(out)[:300]}")
        from io import BytesIO

        from PIL import Image
        pils = [Image.open(BytesIO(SESSION.get(u, timeout=300).content))
                for u in urls]
        info = {k: v for k, v in out.items() if k != "images"}
        print(f"[jz fal image] {len(pils)} image(s) {pils[0].size[0]}x"
              f"{pils[0].size[1]}", flush=True)
        return (pils_to_batch(pils), "\n".join(urls),
                json.dumps(info, ensure_ascii=False)[:4000])


NODE_CLASS_MAPPINGS = {"jz_FalImage": jz_FalImage}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_FalImage": "jz fal Image"}
