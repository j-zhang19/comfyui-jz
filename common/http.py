"""Shared HTTP transport: pooled session + status-aware retries.

Retries 408/429/5xx (honoring Retry-After) and network errors with capped
exponential backoff. Permanent 4xx returns immediately for the caller to
raise — resending a bad request never helps.
"""
import re
import time

import requests

SESSION = requests.Session()

RETRY_BASE_DELAY = 2.0
MAX_ATTEMPTS = 6
BACKOFF_CAP = 45.0
RETRY_AFTER_CAP = 120.0
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def truncate_b64(text: str) -> str:
    """Strip giant base64 blobs out of error bodies before logging/raising."""
    return re.sub(r'"(?:data|url)"\s*:\s*"[A-Za-z0-9+/=:;,]{100,}"',
                  '"data": "<base64 truncated>"', text or "")


def post_with_retries(url: str, headers: dict, payload: dict,
                      timeout: int = 300, tag: str = "jz") -> requests.Response:
    last_err = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            resp = SESSION.post(url, headers=headers, json=payload, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            if attempt < MAX_ATTEMPTS - 1:
                print(f"[{tag}] {type(e).__name__}, retry "
                      f"{attempt + 1}/{MAX_ATTEMPTS - 1}", flush=True)
                time.sleep(min(BACKOFF_CAP, RETRY_BASE_DELAY * 2 ** attempt))
                continue
            break
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS - 1:
            delay = min(BACKOFF_CAP, RETRY_BASE_DELAY * 2 ** attempt)
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = min(float(retry_after), RETRY_AFTER_CAP)
                except ValueError:
                    pass
            print(f"[{tag}] HTTP {resp.status_code}, retry "
                  f"{attempt + 1}/{MAX_ATTEMPTS - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)
            continue
        return resp
    raise RuntimeError(f"{url} unreachable after {MAX_ATTEMPTS} attempts: {last_err}")
