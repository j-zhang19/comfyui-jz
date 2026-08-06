"""JZ OpenRouter VLM — vision/text calls through OpenRouter.

Differences from the vermeer node this replaces:
- real error handling: status-aware retries (408/429/5xx + Retry-After,
  network errors), no retry on permanent 4xx, and failures RAISE with a
  clean message instead of returning an error string downstream
- pooled HTTP session, request timeouts
- images auto-downscaled before upload (OpenRouter 413s past ~10MB)
- the whole IMAGE batch is sent: batch of N frames = N images in the call
- API key: node input > OPENROUTER_API_KEY env > config.ini next to pack
"""
import configparser
import json
import os
from pathlib import Path

from ...common.http import post_with_retries, truncate_b64
from ...common.images import batch_to_data_urls

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_PACK_ROOT = Path(__file__).resolve().parents[2]

MODELS = [
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite-preview",
    "custom",
]


def _resolve_api_key(node_input: str) -> str:
    if node_input.strip():
        return node_input.strip()
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return os.environ["OPENROUTER_API_KEY"].strip()
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(str(_PACK_ROOT / "config.ini"), encoding="utf-8")
    key = cfg.get("API", "OPENROUTER_API_KEY", fallback="").strip()
    if not key:
        raise RuntimeError(
            "No OpenRouter key: set the api_key input, the OPENROUTER_API_KEY "
            "env var, or [API] OPENROUTER_API_KEY in comfyui-jz/config.ini (pack root)")
    return key


class jz_OpenRouterVLM:
    """One call: system instruction + optional text + optional image batch -> text."""

    CATEGORY = "jz/api"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "cost")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {"multiline": True,
                                           "default": "describe this image in detail"}),
                "model": (MODELS, {"default": "google/gemini-3.5-flash"}),
                "max_tokens": ("INT", {"default": 1000, "min": 1, "max": 32768}),
            },
            "optional": {
                "image": ("IMAGE",),
                # forceInput: a connectable socket (pipe text from other
                # nodes), not an inline widget
                "content": ("STRING", {"forceInput": True, "multiline": True,
                                       "default": ""}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "max_edge": ("INT", {"default": 256, "min": 64, "max": 8192,
                                     "tooltip": "images are downscaled to this "
                                                "long edge before upload"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "control_after_generate": True}),
                # appended last (append-only rule). reasoning models (gemini
                # flash, o-series...) silently burn max_tokens on hidden
                # thinking — cap it or answers come back truncated
                "reasoning": (["default", "low", "medium", "high"],
                              {"default": "low"}),
            },
        }

    def run(self, instruction, model, max_tokens, image=None, content="",
            custom_model="", api_key="", max_edge=2048, seed=0,
            reasoning="low"):
        if model == "custom":
            if not custom_model.strip():
                raise RuntimeError("model is 'custom' but custom_model is empty")
            model = custom_model.strip()

        user_parts = []
        if content.strip():
            user_parts.append({"type": "text", "text": content})
        if image is not None:
            for url in batch_to_data_urls(image, max_edge):
                user_parts.append({"type": "image_url", "image_url": {"url": url}})
        if not user_parts:
            user_parts.append({"type": "text", "text": ""})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_parts},
            ],
            "max_tokens": int(max_tokens),
            "seed": int(seed),
            "usage": {"include": True},
        }
        if reasoning != "default":
            payload["reasoning"] = {"effort": reasoning}
        headers = {
            "Authorization": f"Bearer {_resolve_api_key(api_key)}",
            "Content-Type": "application/json",
        }

        resp = post_with_retries(OPENROUTER_URL, headers, payload, tag="jz vlm")
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: "
                               f"{truncate_b64(resp.text)[:400]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: "
                               f"{json.dumps(data['error'], ensure_ascii=False)[:400]}")
        choices = data.get("choices") or []
        if not choices or not (choices[0].get("message") or {}).get("content"):
            raise RuntimeError(f"OpenRouter returned no content: "
                               f"{truncate_b64(json.dumps(data))[:400]}")

        if choices[0].get("finish_reason") == "length":
            print("[jz vlm] WARNING: answer truncated (max_tokens hit — "
                  "reasoning models eat tokens; raise max_tokens or lower "
                  "the reasoning effort)", flush=True)
        text = choices[0]["message"]["content"].strip()
        cost = (data.get("usage") or {}).get("cost")
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "$?"
        return (text, cost_str)


NODE_CLASS_MAPPINGS = {"jz_OpenRouterVLM": jz_OpenRouterVLM}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_OpenRouterVLM": "jz OpenRouter VLM"}
