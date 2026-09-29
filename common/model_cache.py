"""Shared model-list cache for the API nodes: served instantly, refreshed off
the hot path.

INPUT_TYPES() runs at ComfyUI startup AND on every frontend refresh, for every
node. A blocking HTTP call there would stall startup whenever a provider is slow
and break node registration entirely when offline — so nothing here ever blocks
and nothing here ever raises:

    cache fresh (< TTL)  -> use it, no network
    cache stale          -> serve the stale ids now, refresh in the background
    no cache / failure   -> serve the curated list now, refresh in the background

The curated list is always included and always first, so the defaults never move
and a saved workflow's model can't be orphaned by a provider dropping one.
"""
import json
import os
import threading
import time
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = _PACK_ROOT / "models_cache.json"

TTL_SECONDS = 24 * 60 * 60

_MEMO = {}       # kind -> (fetched_ts, ids); the ts matters, see cached_ids
_INFLIGHT = {}   # kind -> True while a background refresh is running
_LOCK = threading.Lock()


def _read_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:  # missing, corrupt, unreadable — all mean "no cache"
        return {}


def _write_cache(kind: str, ids: list) -> None:
    """Atomic: a concurrent reader never sees a half-written file."""
    try:
        cache = _read_cache()
        cache[kind] = {"fetched": time.time(), "ids": ids}
        tmp = CACHE_PATH.with_suffix(".json.tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        os.replace(tmp, CACHE_PATH)
    except Exception:
        pass  # a read-only install still gets working nodes, just no caching


def _refresh(kind: str, fetch) -> None:
    try:
        ids = fetch()
        if ids:
            _write_cache(kind, ids)
            with _LOCK:
                _MEMO[kind] = (time.time(), ids)
    except Exception:
        pass
    finally:
        with _LOCK:
            _INFLIGHT.pop(kind, None)


def _start_refresh(kind: str, fetch) -> None:
    with _LOCK:
        if _INFLIGHT.get(kind):
            return  # one in flight per kind is enough
        _INFLIGHT[kind] = True
    threading.Thread(target=_refresh, args=(kind, fetch),
                     name=f"jz-models-{kind}", daemon=True).start()


def cached_ids(kind: str, curated: list, fetch) -> list:
    """Dropdown options for `kind`. Never blocks on the network, never raises.

    `kind` namespaces the shared cache file, so prefix it with the provider.
    `fetch` is a zero-arg callable returning a list of model ids.
    """
    try:
        with _LOCK:
            memo = _MEMO.get(kind)
        if memo is None:
            entry = _read_cache().get(kind) or {}
            fetched, cached = entry.get("fetched", 0), entry.get("ids") or []
            if cached:
                with _LOCK:
                    _MEMO[kind] = (fetched, cached)
        else:
            # the age has to travel with the memo: reading it from the file
            # only on a cold memo made every warm call look stale and fire a
            # pointless refresh on every INPUT_TYPES()
            fetched, cached = memo
        if time.time() - fetched >= TTL_SECONDS:
            _start_refresh(kind, fetch)
        if not cached:
            return list(curated)
        # curated first (defaults stay put, good models stay visible), then the
        # rest of the catalogue, then the custom escape hatch
        tail = [m for m in cached if m not in curated]
        head = [m for m in curated if m != "custom"]
        return head + tail + (["custom"] if "custom" in curated else [])
    except Exception:
        return list(curated)
