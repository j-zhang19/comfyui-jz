"""jz fal Video — video generation on fal.ai.

Same shape as jz fal Image: a `params` JSON widget validated against the
model's published schema before submitting. Returns a native VIDEO (decoding
a clip into an IMAGE batch costs gigabytes) plus the `request_id`, so a job
you have already paid for can be fetched again rather than re-run.
"""
import json
from io import BytesIO

from comfy_api.latest import InputImpl

from ...common.fal import CURATED_VIDEO, model_ids
from ...common.http import SESSION
from ...common.nodes import scalar
from ._fal_common import run


def _video_url(out: dict) -> str:
    v = out.get("video")
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    if isinstance(v, str):
        return v
    for key in ("videos", "output"):           # a few models name it otherwise
        alt = out.get(key)
        if (isinstance(alt, list) and alt and isinstance(alt[0], dict)
                and alt[0].get("url")):
            return alt[0]["url"]
    raise RuntimeError(f"jz fal Video: no video url in the result: "
                       f"{json.dumps(out)[:300]}")


class jz_FalVideo:
    CATEGORY = "jz/api"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "request_id", "info")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_ids("video", CURATED_VIDEO),
                          {"default": CURATED_VIDEO[0]}),
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a cat walking through grass"}),
                "params": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "model-specific arguments as json, e.g. "
                               '{"duration": "5", "cfg_scale": 0.5}. checked '
                               "against the model's schema before anything is "
                               "billed; the console prints what it accepts"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "start frame / reference, for "
                                               "the image-to-video models"}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "poll_timeout": ("INT", {"default": 1800, "min": 30, "max": 7200}),
                "poll_interval": ("INT", {"default": 5, "min": 1, "max": 30}),
            },
        }

    def generate(self, model, prompt, params, image=None, custom_model="",
                 api_key="", poll_timeout=1800, poll_interval=5):
        model = str(scalar(model, ""))
        if model == "custom":
            custom = str(scalar(custom_model, "")).strip()
            if not custom:
                raise RuntimeError("jz fal Video: model is 'custom' but "
                                   "custom_model is empty")
            model = custom
        rid, out = run(model, prompt, params, image, api_key,
                       poll_timeout, poll_interval, tag="jz fal video")
        url = _video_url(out)
        info = {k: v for k, v in out.items() if k != "video"}
        return (InputImpl.VideoFromFile(BytesIO(SESSION.get(url, timeout=600).content)),
                rid, json.dumps(info, ensure_ascii=False)[:4000])


NODE_CLASS_MAPPINGS = {"jz_FalVideo": jz_FalVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_FalVideo": "jz fal Video"}
