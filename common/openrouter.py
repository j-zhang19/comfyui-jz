"""OpenRouter model lists. The caching lives in common/model_cache.py, shared
with the BytePlus nodes; this module only knows OpenRouter's endpoints.
"""
import json
import urllib.request

from .model_cache import cached_ids

FETCH_TIMEOUT = 5

ENDPOINTS = {
    "chat": "https://openrouter.ai/api/v1/models",
    "image": "https://openrouter.ai/api/v1/images/models",
}


def fetch_ids(kind: str) -> list:
    """Model ids from OpenRouter. Chat is filtered to the vision models, since
    the node that uses it sends images; `custom` still reaches the rest."""
    req = urllib.request.Request(ENDPOINTS[kind],
                                 headers={"User-Agent": "comfyui-jz"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
        data = json.load(r).get("data") or []
    ids = []
    for m in data:
        mid = m.get("id")
        if not mid:
            continue
        if kind == "chat":
            mods = (m.get("architecture") or {}).get("input_modalities") or []
            if "image" not in mods:
                continue
        ids.append(mid)
    return sorted(ids)


def model_ids(kind: str, curated: list) -> list:
    return cached_ids(f"openrouter:{kind}", curated, lambda: fetch_ids(kind))
