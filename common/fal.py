"""Shared fal.ai queue client for the jz fal nodes.

    POST   https://queue.fal.run/{model}          -> {request_id, status_url, response_url}
    GET    {status_url}                            -> IN_QUEUE | IN_PROGRESS | COMPLETED
    GET    {response_url}                          -> the result
    PUT    .../requests/{id}/cancel                -> cancellable

Auth is `Authorization: Key <FAL_KEY>`. The REST is small enough that the
fal_client package is not a dependency.

fal has ~1500 models and every one takes a DIFFERENT input schema, so the
nodes pass model-specific arguments as JSON. What makes that safe is that fal
publishes each model's OpenAPI schema free and unauthenticated, so a typo'd
key, a bad enum or a missing required field is caught HERE, before anything
billable is sent. The same class of mistake on an API without published
schemas has to be discovered by paying for it.

Billable submits are sent once, never auto-retried: a retried POST whose first
attempt actually landed would queue (and bill) a second job.
"""
import json
import urllib.parse
import urllib.request

from .http import SESSION, truncate_b64
from .model_cache import cached_ids
from .secrets import resolve_key

QUEUE = "https://queue.fal.run"
SCHEMA_URL = "https://fal.ai/api/openapi/queue/openapi.json?endpoint_id="
CATALOGUE = "https://fal.ai/api/models?page="
CATALOGUE_PAGES = 5          # 40 per page; the full 1507 would be 38 requests
FETCH_TIMEOUT = 10

IMAGE_CATEGORIES = {"text-to-image", "image-to-image"}
VIDEO_CATEGORIES = {"text-to-video", "image-to-video", "video-to-video"}

CURATED_IMAGE = ["fal-ai/flux/schnell", "fal-ai/flux/dev",
                 "fal-ai/nano-banana-pro", "fal-ai/nano-banana-pro/edit",
                 "custom"]
CURATED_VIDEO = ["fal-ai/kling-video/v2/master/image-to-video",
                 "fal-ai/kling-video/v2/master/text-to-video",
                 "fal-ai/minimax/hailuo-02/standard/image-to-video",
                 "custom"]

_SCHEMAS = {}   # model -> (properties, required); schemas do not change mid-session


def fal_key(node_input: str = "") -> str:
    return resolve_key(
        ["FAL_KEY", "FAL_API_KEY"], ("FAL", "api_key"), node_input,
        "No fal key: set the api_key input, FAL_KEY in comfyui-jz/.env, or "
        "[FAL] api_key in config.ini")


def _headers(key):
    return {"Content-Type": "application/json", "Authorization": f"Key {key}"}


def fetch_ids(kind: str) -> list:
    """Model ids for 'image' or 'video' from the public catalogue."""
    want = IMAGE_CATEGORIES if kind == "image" else VIDEO_CATEGORIES
    out = []
    for page in range(1, CATALOGUE_PAGES + 1):
        req = urllib.request.Request(f"{CATALOGUE}{page}",
                                     headers={"User-Agent": "comfyui-jz"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
            items = json.load(r).get("items") or []
        if not items:
            break
        out += [m["id"] for m in items
                if m.get("id") and not m.get("deprecated")
                and m.get("category") in want]
    return sorted(set(out))


def model_ids(kind: str, curated: list) -> list:
    return cached_ids(f"fal:{kind}", curated, lambda: fetch_ids(kind))


def input_schema(model: str):
    """(properties, required) for a model's queue input, from fal's public
    OpenAPI. Unauthenticated and free — this is what makes validating before
    spending possible."""
    if model in _SCHEMAS:
        return _SCHEMAS[model]
    req = urllib.request.Request(SCHEMA_URL + urllib.parse.quote(model, safe="/"),
                                 headers={"User-Agent": "comfyui-jz"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
        doc = json.load(r)
    schemas = doc.get("components", {}).get("schemas", {})
    name = None
    # prefer the schema the POST path actually references
    post = (doc.get("paths", {}).get(f"/{model}", {}) or {}).get("post", {})
    ref = ((post.get("requestBody", {}).get("content", {})
            .get("application/json", {}).get("schema", {})) or {}).get("$ref", "")
    if ref:
        name = ref.rsplit("/", 1)[-1]
    if name not in schemas:
        name = next((n for n in schemas if n.endswith("Input")), None)
    if not name:
        raise RuntimeError(f"no input schema for {model}")
    s = schemas[name]
    found = (s.get("properties") or {}, list(s.get("required") or []))
    _SCHEMAS[model] = found
    return found


def _allowed(spec: dict) -> list:
    """Enum values for a property, looking through anyOf branches."""
    if "enum" in spec:
        return list(spec["enum"])
    for branch in spec.get("anyOf") or []:
        if "enum" in branch:
            return list(branch["enum"])
    return []


_PY = {"string": str, "integer": int, "number": (int, float),
       "boolean": bool, "array": list, "object": dict}


def _type_ok(spec: dict, value) -> bool:
    branches = spec.get("anyOf") or [spec]
    types = [b.get("type") for b in branches if b.get("type")]
    if not types or any(t == "null" for t in types) and value is None:
        return True
    if any("$ref" in b for b in branches):
        return True  # nested model — let the server judge
    for t in types:
        py = _PY.get(t)
        if py is None:
            return True
        if t == "integer" and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def validate_args(model: str, args: dict) -> list:
    """Raise on anything the model will reject. Returns the property names."""
    try:
        props, required = input_schema(model)
    except Exception as e:
        print(f"[jz fal] could not fetch the schema for {model} ({e}) — "
              f"submitting unvalidated; fal will still reject bad arguments "
              f"with a 422 before charging", flush=True)
        return []
    missing = [r for r in required if r not in args or args[r] in ("", None)]
    if missing:
        raise ValueError(f"jz fal: {model} requires {', '.join(missing)} — "
                         f"accepted parameters: {', '.join(sorted(props))}")
    for k, v in args.items():
        if k not in props:
            raise ValueError(f"jz fal: {model} has no parameter {k!r} — "
                             f"accepted: {', '.join(sorted(props))}")
        allowed = _allowed(props[k])
        if allowed and v not in allowed:
            raise ValueError(f"jz fal: {k}={v!r} is not valid for {model} — "
                             f"allowed: {', '.join(map(str, allowed))}")
        if not _type_ok(props[k], v):
            raise ValueError(f"jz fal: {k} should be "
                             f"{props[k].get('type') or 'another type'}, "
                             f"got {type(v).__name__}")
    return sorted(props)


def _check(r):
    if r.status_code >= 400:
        raise RuntimeError(f"fal HTTP {r.status_code}: {truncate_b64(r.text)[:400]}")
    return r.json()


def submit(model, args, key):
    """Billable. Sent once — a retry whose first attempt landed bills twice."""
    return _check(SESSION.post(f"{QUEUE}/{model}", headers=_headers(key),
                               json=args, timeout=120))


def status(url, key):
    return _check(SESSION.get(url, headers=_headers(key), timeout=60))


def result(url, key):
    return _check(SESSION.get(url, headers=_headers(key), timeout=300))


def cancel(model, request_id, key):
    """fal can cancel a queued/running job — worth doing on timeout."""
    try:
        SESSION.put(f"{QUEUE}/{model}/requests/{request_id}/cancel",
                    headers=_headers(key), timeout=30)
        return True
    except Exception:
        return False
