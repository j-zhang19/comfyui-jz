"""Model lists for the OpenRouter nodes: cached, refreshed off the hot path.

INPUT_TYPES() runs at ComfyUI startup AND on every frontend refresh, for every
node. A blocking HTTP call there would stall startup whenever OpenRouter is slow
and break node registration entirely when offline — so nothing here ever blocks
and nothing here ever raises:

    cache fresh (< TTL)  -> use it, no network
    cache stale          -> serve the stale ids now, refresh in the background
    no cache / failure   -> serve the curated list now, refresh in the background

The curated list is always included and always first, so the defaults never move
and a saved workflow's model can't be orphaned by OpenRouter dropping it.
"""
import json
import os
import threading
import time
import urllib.request
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parents[1]
_CACHE_PATH = _PACK_ROOT / "models_cache.json"

TTL_SECONDS = 24 * 60 * 60
FETCH_TIMEOUT = 5

ENDPOINTS = {
    "chat": "https://openrouter.ai/api/v1/models",
    "image": "https://openrouter.ai/api/v1/images/models",
}

_MEMO = {}       # kind -> ids, so repeated INPUT_TYPES calls skip the file read
_INFLIGHT = {}   # kind -> True while a background refresh is running
_LOCK = threading.Lock()


def _fetch_ids(kind: str) -> list:
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


def _read_cache() -> dict:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # missing, corrupt, unreadable — all mean "no cache"
        return {}


def _write_cache(kind: str, ids: list) -> None:
    """Atomic: a concurrent reader never sees a half-written file."""
    try:
        cache = _read_cache()
        cache[kind] = {"fetched": time.time(), "ids": ids}
        tmp = _CACHE_PATH.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, _CACHE_PATH)
    except Exception:
        pass  # a read-only install still gets working nodes, just no caching


def _refresh(kind: str) -> None:
    try:
        ids = _fetch_ids(kind)
        if ids:
            _write_cache(kind, ids)
            with _LOCK:
                _MEMO[kind] = ids
    except Exception:
        pass
    finally:
        with _LOCK:
            _INFLIGHT.pop(kind, None)


def _start_refresh(kind: str) -> None:
    with _LOCK:
        if _INFLIGHT.get(kind):
            return  # one in flight per kind is enough
        _INFLIGHT[kind] = True
    threading.Thread(target=_refresh, args=(kind,),
                     name=f"jz-or-models-{kind}", daemon=True).start()


def model_ids(kind: str, curated: list) -> list:
    """Dropdown options for `kind`. Never blocks on the network, never raises."""
    try:
        with _LOCK:
            cached = _MEMO.get(kind)
        entry = {} if cached is not None else (_read_cache().get(kind) or {})
        if cached is None:
            cached = entry.get("ids") or []
            if cached:
                with _LOCK:
                    _MEMO[kind] = cached
        fresh = bool(entry) and (time.time() - entry.get("fetched", 0)) < TTL_SECONDS
        if not fresh:
            _start_refresh(kind)
        if not cached:
            return list(curated)
        # curated first (defaults stay put, good models stay visible), then the
        # rest of the catalogue, then the custom escape hatch
        tail = [m for m in cached if m not in curated]
        head = [m for m in curated if m != "custom"]
        return head + tail + (["custom"] if "custom" in curated else [])
    except Exception:
        return list(curated)
