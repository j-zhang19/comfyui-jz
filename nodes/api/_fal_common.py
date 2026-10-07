"""Shared body of the two jz fal nodes: build args, submit, poll, fetch.

Kept out of the node files because image and video differ only in how they
turn the finished result into a ComfyUI output.
"""
import json
import time

from ...common.fal import (cancel, fal_key, input_schema, result, status,
                           submit, validate_args)
from ...common.images import batch_to_data_urls
from ...common.nodes import scalar


def image_uris(image) -> list:
    """Batch tensor or LIST of them -> data URIs, so mixed sizes work."""
    if image is None:
        return []
    out = []
    for t in (image if isinstance(image, list) else [image]):
        if t is not None:
            out += batch_to_data_urls(t)
    return out


def build_args(model, prompt, params, image):
    """prompt + parsed JSON params + images, wired by what the schema declares."""
    try:
        args = json.loads(params) if str(params).strip() else {}
    except json.JSONDecodeError as e:
        raise ValueError(f"jz fal: params is not valid json — {e}") from e
    if not isinstance(args, dict):
        raise ValueError(f"jz fal: params must be a json object, got "
                         f"{type(args).__name__}")
    if prompt.strip():
        args.setdefault("prompt", prompt.strip())

    uris = image_uris(image)
    if uris:
        try:
            props, _ = input_schema(model)
        except Exception:
            props = {}
        # the schema says whether this model takes many references, one, or
        # none at all — a text-to-image model must say so rather than be sent
        # an argument it will reject
        if "image_urls" in props:
            args.setdefault("image_urls", uris)
        elif "image_url" in props:
            args.setdefault("image_url", uris[0])
            if len(uris) > 1:
                print(f"[jz fal] {model} takes a single image_url — using the "
                      f"first of {len(uris)}", flush=True)
        elif not props:
            args.setdefault("image_url", uris[0])  # schema unavailable; try
        else:
            raise ValueError(
                f"jz fal: {model} takes no image input — it accepts "
                f"{', '.join(sorted(props))}. use an edit / image-to-image / "
                f"image-to-video model, or unwire the image")
    return args


def run(model, prompt, params, image, api_key, poll_timeout, poll_interval,
        tag="jz fal"):
    """Validate, submit once, poll, return the finished result dict."""
    model = str(scalar(model, "")).strip()
    args = build_args(model, str(scalar(prompt, "")), scalar(params, "{}"), image)
    accepted = validate_args(model, args)          # raises BEFORE spending
    if accepted:
        print(f"[{tag}] {model} accepts: {', '.join(accepted)}", flush=True)

    key = fal_key(str(scalar(api_key, "")))
    print(f"[{tag}] submitting {model} with "
          f"{', '.join(sorted(args))} — billed from here", flush=True)
    queued = submit(model, args, key)
    rid = queued.get("request_id")
    status_url, response_url = queued.get("status_url"), queued.get("response_url")
    if not (rid and status_url and response_url):
        raise RuntimeError(f"jz fal: unexpected submit response: "
                           f"{json.dumps(queued)[:300]}")
    print(f"[{tag}] request {rid}", flush=True)

    timeout = int(scalar(poll_timeout, 900))
    interval = int(scalar(poll_interval, 3))
    deadline, last = time.time() + timeout, ""
    while time.time() < deadline:
        try:
            st = status(status_url, key).get("status", "")
        except Exception as e:                    # reads are free, ride it out
            print(f"[{tag}] poll error (retrying): {e}", flush=True)
            time.sleep(interval)
            continue
        if st != last:
            print(f"[{tag}] {rid} {st}", flush=True)
            last = st
        if st == "COMPLETED":
            return rid, result(response_url, key)
        time.sleep(interval)

    # fal can cancel, unlike some queues — do not leave it running and billing
    killed = cancel(model, rid, key)
    raise RuntimeError(
        f"jz fal: {model} timed out after {timeout}s; "
        f"{'cancelled' if killed else 'COULD NOT CANCEL'} request {rid}")
