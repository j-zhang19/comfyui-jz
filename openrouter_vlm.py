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
import base64
import configparser
import io
import json
import os
import re
import time

import numpy as np
import requests
import torch
from PIL import Image

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

_SESSION = requests.Session()
_RETRY_BASE_DELAY = 2.0
_MAX_ATTEMPTS = 6
_BACKOFF_CAP = 45.0
_RETRY_AFTER_CAP = 120.0
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

MODELS = [
    "anthropic/claude-opus-4.8",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite-preview",
    "custom",
]


def _truncate_b64(text: str) -> str:
    return re.sub(r'"(?:data|url)"\s*:\s*"[A-Za-z0-9+/=:;,]{100,}"',
                  '"data": "<base64 truncated>"', text or "")


def _resolve_api_key(node_input: str) -> str:
    if node_input.strip():
        return node_input.strip()
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return os.environ["OPENROUTER_API_KEY"].strip()
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(os.path.join(os.path.dirname(__file__), "config.ini"), encoding="utf-8")
    key = cfg.get("API", "OPENROUTER_API_KEY", fallback="").strip()
    if not key:
        raise RuntimeError(
            "No OpenRouter key: set the api_key input, the OPENROUTER_API_KEY "
            "env var, or [API] OPENROUTER_API_KEY in comfyui-jz/config.ini")
    return key


def _post_with_retries(payload: dict, headers: dict) -> requests.Response:
    last_err = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = _SESSION.post(OPENROUTER_URL, headers=headers, json=payload,
                                 timeout=300)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < _MAX_ATTEMPTS - 1:
                print(f"[JZ VLM] {type(e).__name__}, retry "
                      f"{attempt + 1}/{_MAX_ATTEMPTS - 1}", flush=True)
                time.sleep(min(_BACKOFF_CAP, _RETRY_BASE_DELAY * 2 ** attempt))
                continue
            break
        if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_ATTEMPTS - 1:
            delay = min(_BACKOFF_CAP, _RETRY_BASE_DELAY * 2 ** attempt)
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = min(float(retry_after), _RETRY_AFTER_CAP)
                except ValueError:
                    pass
            print(f"[JZ VLM] HTTP {resp.status_code}, retry "
                  f"{attempt + 1}/{_MAX_ATTEMPTS - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
            continue
        return resp
    raise RuntimeError(f"OpenRouter unreachable after {_MAX_ATTEMPTS} attempts: {last_err}")


def _batch_to_data_urls(image: torch.Tensor, max_edge: int) -> list[str]:
    """Every frame of the BHWC batch becomes one image part."""
    urls = []
    for frame in image:
        arr = (frame.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr)
        if max(pil.size) > max_edge:
            scale = max_edge / max(pil.size)
            pil = pil.resize((max(1, round(pil.width * scale)),
                              max(1, round(pil.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="PNG")
        urls.append("data:image/png;base64,"
                    + base64.b64encode(buf.getvalue()).decode())
    return urls


class JZ_OpenRouterVLM:
    """One call: system instruction + optional text + optional image batch -> text."""

    CATEGORY = "JZ/llm"
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
                "content": ("STRING", {"multiline": True, "default": ""}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "max_edge": ("INT", {"default": 2048, "min": 256, "max": 8192,
                                     "tooltip": "images are downscaled to this "
                                                "long edge before upload"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def run(self, instruction, model, max_tokens, image=None, content="",
            custom_model="", api_key="", max_edge=2048, seed=0):
        if model == "custom":
            if not custom_model.strip():
                raise RuntimeError("model is 'custom' but custom_model is empty")
            model = custom_model.strip()

        user_parts = []
        if content.strip():
            user_parts.append({"type": "text", "text": content})
        if image is not None:
            for url in _batch_to_data_urls(image, max_edge):
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
        headers = {
            "Authorization": f"Bearer {_resolve_api_key(api_key)}",
            "Content-Type": "application/json",
        }

        resp = _post_with_retries(payload, headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenRouter HTTP {resp.status_code}: "
                               f"{_truncate_b64(resp.text)[:400]}")
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error: "
                               f"{json.dumps(data['error'], ensure_ascii=False)[:400]}")
        choices = data.get("choices") or []
        if not choices or not (choices[0].get("message") or {}).get("content"):
            raise RuntimeError(f"OpenRouter returned no content: "
                               f"{_truncate_b64(json.dumps(data))[:400]}")

        text = choices[0]["message"]["content"].strip()
        cost = (data.get("usage") or {}).get("cost")
        cost_str = f"${cost:.4f}" if isinstance(cost, (int, float)) else "$?"
        return (text, cost_str)


NODE_CLASS_MAPPINGS = {"JZ_OpenRouterVLM": JZ_OpenRouterVLM}
NODE_DISPLAY_NAME_MAPPINGS = {"JZ_OpenRouterVLM": "JZ OpenRouter VLM"}
