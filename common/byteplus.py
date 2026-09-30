"""Shared BytePlus ModelArk (Ark) client for the Seedream / Seedance nodes.

Base https://ark.ap-southeast.bytepluses.com/api/v3, bearer auth, key from
BYTEPLUS_API_KEY (or ARK_API_KEY, or config.ini [BYTEDANCE]).

    image  POST /images/generations          synchronous, ~8s
    video  POST /contents/generations/tasks  async, poll GET .../tasks/{id}

Two properties of this API drive everything here, both learned the expensive
way:

1. The video endpoint SILENTLY IGNORES unknown flags and invalid flag values
   and bills you anyway. Only --resolution and --duration are checked server
   side; a typo'd --ratio/--fps/--wm/--cf/--seed produces a perfectly valid,
   perfectly wrong, fully charged video. So every flag is whitelisted here
   BEFORE anything is sent, and the caller reports the parameters the server
   echoes back so a dropped flag is visible instead of silent.

2. A running task CANNOT be cancelled — DELETE returns 409
   InvalidAction.RunningTaskDeletion. The charge is committed the moment the
   POST returns 200. That is why billable POSTs here are sent ONCE, with no
   automatic retries: a retried submit whose first attempt actually landed
   would create a second, separately billed task. Polling GETs are idempotent
   and do retry.
"""
import json

from .http import SESSION, truncate_b64
from .images import batch_to_data_urls
from .model_cache import cached_ids
from .nodes import format_usd
from .secrets import byteplus_key

REGIONS = {
    "ap-southeast": "https://ark.ap-southeast.bytepluses.com/api/v3",
    "eu-west": "https://ark.eu-west.bytepluses.com/api/v3",
}

# task_type is the reliable discriminator: `modalities` is incomplete upstream
# (dola-seedream-5-0-flash-260915 is live with no output_modalities at all)
IMAGE_TASKS = {"TextToImage", "ImageToImage"}
VIDEO_TASKS = {"TextToVideo", "ImageToVideo", "MultimodalToVideo",
               "VideoEditing", "VideoExtension"}

CURATED_IMAGE = ["seedream-4-0-250828", "seedream-4-5-251128",
                 "seedream-5-0-260128", "dola-seedream-5-0-pro-260628", "custom"]
CURATED_VIDEO = ["seedance-1-0-pro-250528", "seedance-1-0-pro-fast-251015",
                 "dreamina-seedance-2-5-260628", "dreamina-seedance-2-0-mini-260615",
                 "custom"]

RESOLUTIONS = ["480p", "720p", "1080p", "4k"]
RATIOS = ["16:9", "9:16", "4:3", "3:4", "1:1", "21:9", "9:21", "adaptive"]
IMAGE_SIZES = ["1k", "2k", "4k", "custom"]

# $ per 1M tokens on Ark. Matched by prefix because ids carry a date suffix.
TOKEN_RATES = {
    "seedance-1-0-pro-fast": 1.00,
    "seedance-1-0-pro": 2.50,
}

FETCH_TIMEOUT = 8


def _headers(key):
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def fetch_ids(kind: str) -> list:
    """Live model ids for 'image' or 'video'. Needs the key; on any failure the
    caller falls back to the curated list."""
    want = IMAGE_TASKS if kind == "image" else VIDEO_TASKS
    r = SESSION.get(f"{REGIONS['ap-southeast']}/models",
                    headers=_headers(byteplus_key()), timeout=FETCH_TIMEOUT)
    r.raise_for_status()
    out = []
    for m in r.json().get("data") or []:
        if m.get("status") == "Shutdown":
            continue
        if set(m.get("task_type") or []) & want:
            out.append(m["id"])
    return sorted(out)


def model_ids(kind: str, curated: list) -> list:
    return cached_ids(f"byteplus:{kind}", curated, lambda: fetch_ids(kind))


def resolve_model(model: str, custom_model: str) -> str:
    if model != "custom":
        return model
    if not custom_model.strip():
        raise RuntimeError("model is 'custom' but custom_model is empty")
    return custom_model.strip()


def _raise_http(r, region):
    """BytePlus says "the API key ... is missing or invalid" for a key that is
    simply issued for the other region, which reads as a key problem when it is
    a region problem. Say so."""
    if r.status_code == 401:
        raise RuntimeError(
            f"BytePlus 401 in region '{region}': the key was rejected. Keys are "
            f"REGION-SCOPED — an {'/'.join(REGIONS)} key fails on the other "
            f"region. Check the `region` widget matches the key, and that the "
            f"key is set (api_key input, BYTEPLUS_API_KEY in comfyui-jz/.env, "
            f"or [BYTEDANCE] ARK_API_KEY in config.ini). "
            f"{truncate_b64(r.text)[:200]}")
    raise RuntimeError(f"BytePlus HTTP {r.status_code}: "
                       f"{truncate_b64(r.text)[:400]}")


def api_post(region, path, payload, key, timeout=600):
    """A billable submit: sent ONCE. See the module docstring — a retry whose
    first attempt actually landed would bill twice and cannot be cancelled."""
    r = SESSION.post(f"{REGIONS[region]}{path}", headers=_headers(key),
                     json=payload, timeout=timeout)
    if r.status_code >= 400:
        _raise_http(r, region)
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"BytePlus error: "
                           f"{json.dumps(data['error'], ensure_ascii=False)[:400]}")
    return data


def api_get(region, path, key, timeout=60):
    """Idempotent read — safe to call repeatedly."""
    r = SESSION.get(f"{REGIONS[region]}{path}", headers=_headers(key),
                    timeout=timeout)
    if r.status_code >= 400:
        _raise_http(r, region)
    return r.json()


def image_refs(image) -> list:
    """Every frame becomes one reference data-URI.

    Accepts a batch tensor or a LIST of them (what jz Resize Long Edge emits),
    so mixed-size references work — a tensor cannot hold mixed sizes but the
    API takes each reference separately. Sent at native size: unlike
    OpenRouter, Ark does not 413 on large refs.
    """
    if image is None:
        return []
    out = []
    for t in (image if isinstance(image, list) else [image]):
        if t is not None:
            out += batch_to_data_urls(t)
    return out


def build_flags(resolution="auto", ratio="auto", duration=0, fps=0,
                camera_fixed=None, watermark=None, seed=-1) -> str:
    """Seedance parameters ride as text flags on the prompt. Validated here
    because the server does not validate most of them — it just ignores a bad
    value and charges full price for the wrong video."""
    parts = []
    if resolution and resolution != "auto":
        if resolution not in RESOLUTIONS:
            raise ValueError(f"resolution {resolution!r} is not one of "
                             f"{', '.join(RESOLUTIONS)}")
        parts.append(f"--rs {resolution}")
    if ratio and ratio != "auto":
        if ratio not in RATIOS:
            raise ValueError(f"ratio {ratio!r} is not one of {', '.join(RATIOS)}")
        parts.append(f"--rt {ratio}")
    if duration:
        if not 1 <= int(duration) <= 60:
            raise ValueError(f"duration {duration} is outside 1-60 seconds")
        parts.append(f"--dur {int(duration)}")
    if fps:
        if not 1 <= int(fps) <= 60:
            raise ValueError(f"fps {fps} is outside 1-60")
        parts.append(f"--fps {int(fps)}")
    if camera_fixed is not None:
        parts.append(f"--cf {'true' if camera_fixed else 'false'}")
    if watermark is not None:
        parts.append(f"--wm {'true' if watermark else 'false'}")
    if seed is not None and int(seed) >= 0:
        parts.append(f"--seed {int(seed)}")
    return " ".join(parts)


_SHORT_EDGE = {"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160}


def estimate_tokens(resolution, ratio, duration, fps):
    """Ark bills video by tokens = w*h*fps*seconds/1024, with each side rounded
    to a multiple of 16. Verified exactly against a real 1080p/16:9/5s/24fps
    job: 1936x1088 -> 246,840 tokens."""
    if resolution not in _SHORT_EDGE or not ratio or ":" not in ratio:
        return None
    if not duration or not fps:
        return None
    short = -(-_SHORT_EDGE[resolution] // 16) * 16
    a, b = (int(x) for x in ratio.split(":"))
    long_ = int(round(short * max(a, b) / min(a, b) / 16) * 16)
    return int(long_ * short * int(fps) * int(duration) / 1024)


def token_rate(model: str):
    for prefix, rate in TOKEN_RATES.items():
        if model.startswith(prefix):
            return rate
    return None


def format_cost(model, tokens):
    rate = token_rate(model)
    return format_usd(tokens / 1_000_000 * rate if tokens and rate else None)
