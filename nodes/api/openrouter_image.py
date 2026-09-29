"""jz OpenRouter Image — image generation and editing through OpenRouter.

A separate node from jz OpenRouter VLM because it is a different API, not a
mode of the same one: POST /api/v1/images instead of /chat/completions, a
prompt/n/aspect_ratio/resolution body instead of messages, images returned as
data[].b64_json instead of choices[].message.content, and its own model
catalogue at /api/v1/images/models that does not overlap /api/v1/models.

Wire an IMAGE in and every frame becomes an `input_references` entry, which is
how editing / img2img works here.

Supported parameters vary sharply per model — `resolution` does not exist on
gpt-5-image or flux.2-pro, `n` maxes at 1 for most models but 10 for
gpt-5-image, `seed` is unsupported on gemini-3-pro-image. So `auto` on a widget
omits the field entirely rather than sending a default the model may reject.
"""
import base64
import io
import json

from PIL import Image

from ...common.http import post_with_retries, truncate_b64
from ...common.images import batch_to_data_urls, pils_to_batch
from ...common.nodes import format_usd
from ...common.openrouter import model_ids
from ...common.secrets import openrouter_key

OPENROUTER_IMAGE_URL = "https://openrouter.ai/api/v1/images"

# curated favourites, always first in the dropdown. model_ids() appends the
# rest of the live catalogue from a background-refreshed cache, and falls back
# to exactly this list when there is none.
CURATED = [
    "google/gemini-3-pro-image",
    "google/gemini-3.1-flash-image",
    "google/gemini-3.1-flash-lite-image",
    "bytedance-seed/seedream-5-0-pro",
    "bytedance-seed/seedream-5-0-lite",
    "black-forest-labs/flux.2-pro",
    "openai/gpt-5-image",
    "openai/gpt-5-image-mini",
    "custom",
]

# the endpoint's own enum, read off its schema validation
ASPECT_RATIOS = [
    "auto", "1:1", "1:2", "1:4", "1:8", "2:1", "2:3", "2.35:1", "3:2", "3:4",
    "4:1", "4:3", "4:5", "5:2", "5:4", "8:1", "9:16", "16:9", "9:19.5",
    "19.5:9", "9:20", "20:9", "9:21", "21:9",
]
RESOLUTIONS = ["auto", "512", "1K", "2K", "4K"]


def decode_image(item: dict) -> Image.Image:
    """One data[] entry -> PIL. b64_json is the documented field; a url form is
    tolerated because providers differ and this path is hard to exercise."""
    b64 = item.get("b64_json")
    if not b64:
        url = item.get("url") or (item.get("image_url") or {}).get("url") or ""
        if url.startswith("data:"):
            b64 = url.split(",", 1)[-1]
    if not b64:
        raise RuntimeError(
            f"jz OpenRouter Image: no image bytes in a result entry "
            f"(keys: {sorted(item)})")
    return Image.open(io.BytesIO(base64.b64decode(b64)))


class jz_OpenRouterImage:
    """One call: prompt + optional reference images -> an IMAGE batch."""

    CATEGORY = "jz/api"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("images", "cost", "usage")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a photo of a cute dog"}),
                "model": (model_ids("image", CURATED), {"default": CURATED[0]}),
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "auto",
                    "tooltip": "auto omits the field — models support "
                               "different subsets of these"}),
                "resolution": (RESOLUTIONS, {
                    "default": "auto",
                    "tooltip": "auto omits the field; gpt-5-image and flux "
                               "have no resolution parameter at all"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10,
                              "tooltip": "most models cap this at 1; "
                                         "gpt-5-image allows up to 10"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "control_after_generate": True,
                                 "tooltip": "sent only when non-zero — "
                                            "unsupported on some models"}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "reference images for editing / img2img — every "
                               "frame of the batch becomes one reference"}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "max_edge": ("INT", {"default": 2048, "min": 64, "max": 8192,
                                     "tooltip": "references are downscaled to "
                                                "this long edge before upload"}),
            },
        }

    def generate(self, prompt, model, aspect_ratio, resolution, n, seed,
                 image=None, custom_model="", api_key="", max_edge=2048):
        if model == "custom":
            if not custom_model.strip():
                raise RuntimeError("jz OpenRouter Image: model is 'custom' but "
                                   "custom_model is empty")
            model = custom_model.strip()
        if not prompt.strip():
            raise ValueError("jz OpenRouter Image: prompt is empty")

        # only send what was explicitly chosen: an unsupported parameter is a
        # 400 from the provider, and every model supports a different subset
        payload = {"model": model, "prompt": prompt, "n": int(n)}
        if aspect_ratio != "auto":
            payload["aspect_ratio"] = aspect_ratio
        if resolution != "auto":
            payload["resolution"] = resolution
        if int(seed):
            payload["seed"] = int(seed)
        if image is not None:
            payload["input_references"] = [
                {"type": "image_url", "image_url": {"url": url}}
                for url in batch_to_data_urls(image, max_edge)
            ]

        headers = {"Authorization": f"Bearer {openrouter_key(api_key)}",
                   "Content-Type": "application/json"}
        resp = post_with_retries(OPENROUTER_IMAGE_URL, headers, payload,
                                 timeout=600, tag="jz or-image")
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: "
                               f"{truncate_b64(resp.text)[:400]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: "
                               f"{json.dumps(data['error'], ensure_ascii=False)[:400]}")
        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"OpenRouter returned no images: "
                               f"{truncate_b64(json.dumps(data))[:400]}")

        batch = pils_to_batch([decode_image(it) for it in items])
        usage = data.get("usage") or {}
        return (batch, format_usd(usage.get("cost")),
                json.dumps(usage, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {"jz_OpenRouterImage": jz_OpenRouterImage}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_OpenRouterImage": "jz OpenRouter Image"}
