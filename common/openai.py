"""Shared OpenAI client for jz OpenAI Image.

Base https://api.openai.com/v1, bearer auth, key from OPENAI_API_KEY (or
config.ini [OPENAI]).

    POST /images/generations, /images/edits   synchronous

No video: OpenAI shut down Sora and the whole /v1/videos API on 2026-09-24
with no replacement (the routes now 404).

No idempotency key exists on these endpoints, so a retried POST whose first
attempt landed is a second, separately billed generation. Billable POSTs are
therefore sent ONCE.
"""
import json

from .http import SESSION, truncate_b64
from .images import batch_to_data_urls
from .model_cache import cached_ids
from .nodes import format_usd
from .secrets import openai_key

BASE = "https://api.openai.com/v1"
FETCH_TIMEOUT = 8

CURATED_IMAGE = ["gpt-image-2.5-sunburst", "gpt-image-2.5-flare", "gpt-image-2",
                 "custom"]

# $ per 1M tokens: (text in, image in, text out, image out). Direct /images
# calls never get the cached-input rate. Longest prefix wins, because snapshot
# ids carry a date suffix and gpt-image-2 is a prefix of gpt-image-2.5-*.
IMAGE_RATES = {
    "gpt-image-2.5-sunburst": (5.00, 8.00, 0.0, 30.00),
    "gpt-image-2.5-flare": (5.00, 8.00, 0.0, 30.00),
    "gpt-image-2": (5.00, 8.00, 0.0, 30.00),
    "gpt-image-1.5": (5.00, 8.00, 10.00, 32.00),
    "gpt-image-1-mini": (2.00, 2.50, 0.0, 8.00),
    "gpt-image-1": (5.00, 10.00, 0.0, 40.00),
}


def _headers(key):
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def fetch_ids(prefix: str) -> list:
    """Live model ids starting with `prefix`. Needs the key; on any failure the
    caller falls back to the curated list."""
    r = SESSION.get(f"{BASE}/models", headers=_headers(openai_key()),
                    timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data") or []
                  if str(m.get("id", "")).startswith(prefix))


def model_ids(kind: str, prefix: str, curated: list) -> list:
    return cached_ids(f"openai:{kind}", curated, lambda: fetch_ids(prefix))


def resolve_model(model: str, custom_model: str) -> str:
    if model != "custom":
        return model
    if not custom_model.strip():
        raise RuntimeError("model is 'custom' but custom_model is empty")
    return custom_model.strip()


def _raise_http(r):
    """Turn an OpenAI error body into one readable line, with the two failures
    that read as something else spelled out."""
    try:
        err = r.json().get("error") or {}
    except ValueError:
        err = {}
    msg = err.get("message") or truncate_b64(r.text)[:400]
    code = err.get("code") or err.get("type") or ""
    rid = r.headers.get("x-request-id", "")
    hint = ""
    if r.status_code == 403 and "verified" in msg:
        hint = (" -> verify the organization at platform.openai.com/settings/"
                "organization/general (can take ~15 min to propagate)")
    elif code == "moderation_blocked":
        details = err.get("moderation_details") or {}
        hint = (f" -> blocked at {details.get('moderation_stage', '?')} stage "
                f"{details.get('categories') or ''}")
    elif code == "insufficient_quota":
        hint = " -> out of credit, not a rate limit; retrying will not help"
    raise RuntimeError(f"OpenAI HTTP {r.status_code} {code}: {msg}{hint}"
                       f"{f' [request {rid}]' if rid else ''}")


def api_post(path, payload, key, timeout=600):
    """A billable submit: sent ONCE (see module docstring)."""
    r = SESSION.post(f"{BASE}{path}", headers=_headers(key), json=payload,
                     timeout=(10, timeout))
    if r.status_code >= 400:
        _raise_http(r)
    return r.json()


def image_refs(image) -> list:
    """Every frame becomes one data-URL reference. Takes a batch or a LIST of
    batches (jz Resize Long Edge), so mixed-size references work."""
    out = []
    for t in (image if isinstance(image, list) else [image]):
        if t is not None:
            out += batch_to_data_urls(t)
    return out


def _rate(model: str, table: dict):
    for prefix in sorted(table, key=len, reverse=True):
        if model.startswith(prefix):
            return table[prefix]
    return None


def image_cost(model: str, usage: dict) -> str:
    """USD from the usage block. Falls back to all output as image tokens when
    the per-modality breakdown is missing."""
    rate = _rate(model, IMAGE_RATES)
    if not rate or not usage:
        return format_usd(None)
    text_in, image_in, text_out, image_out = rate
    ind = usage.get("input_tokens_details") or {}
    outd = usage.get("output_tokens_details")
    if outd is None:
        outd = {"image_tokens": usage.get("output_tokens", 0)}
    if not ind:
        ind = {"text_tokens": usage.get("input_tokens", 0)}
    usd = (ind.get("text_tokens", 0) * text_in
           + ind.get("image_tokens", 0) * image_in
           + outd.get("text_tokens", 0) * text_out
           + outd.get("image_tokens", 0) * image_out) / 1e6
    return format_usd(usd)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
