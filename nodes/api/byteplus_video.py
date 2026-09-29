"""jz BytePlus Seedance — video generation on BytePlus ModelArk.

Two nodes:
  jz BytePlus Seedance        submit + poll + download, returns a VIDEO
  jz BytePlus Seedance Fetch  pick up a task by id (tasks live 48h)

The generate node also outputs `task_id`, so a job you have already paid for
can be re-fetched for free if the graph errors after the spend.

SAFETY. Seedance parameters ride as text flags on the prompt, and the server
validates only --resolution and --duration; a typo'd --ratio/--fps/--wm/--cf/
--seed is silently ignored and the wrong video is billed in full. A running
task cannot be cancelled (DELETE -> 409). So:

  - every flag is whitelisted client side before the submit (common/byteplus)
  - the token cost is estimated and printed BEFORE spending
  - `applied` returns the parameters the server echoes back, so a flag that
    did not take is visible instead of silent
  - `seed` defaults to -1 and is never randomised, so re-queuing a graph
    cannot quietly start another billable job
"""
import io
import json
import time

from comfy_api.latest import InputImpl

from ...common.byteplus import (CURATED_VIDEO, RATIOS, REGIONS, RESOLUTIONS,
                                api_get, api_post, build_flags,
                                estimate_tokens, format_cost, image_refs,
                                model_ids, resolve_model)
from ...common.secrets import byteplus_key
from ...common.http import SESSION

TASKS = "/contents/generations/tasks"
ECHOED = ("resolution", "ratio", "duration", "framespersecond", "seed",
          "output_format")


def _applied(task: dict) -> str:
    """What the server actually used — the check against a silently dropped flag."""
    return json.dumps({k: task[k] for k in ECHOED if k in task},
                      ensure_ascii=False)


def _download(url: str):
    return InputImpl.VideoFromFile(io.BytesIO(SESSION.get(url, timeout=600).content))


def _await(region, key, task_id, timeout, interval, tag="jz seedance"):
    """Poll until the task settles. Reads are free, so retry through errors."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            task = api_get(region, f"{TASKS}/{task_id}", key)
        except Exception as e:  # transient — a paid task is worth waiting out
            print(f"[{tag}] poll error (retrying): {e}", flush=True)
            time.sleep(interval)
            continue
        status = task.get("status")
        if status != last:
            print(f"[{tag}] {task_id} {status}", flush=True)
            last = status
        if status == "succeeded":
            return task
        if status == "failed":
            raise RuntimeError(
                f"jz BytePlus Seedance: task {task_id} failed: "
                f"{json.dumps(task.get('error') or task)[:300]}")
        time.sleep(interval)
    raise RuntimeError(
        f"jz BytePlus Seedance: timed out after {timeout}s. The task is still "
        f"running and already paid for — fetch it later with task_id "
        f"{task_id} (tasks live 48h)")


def _finish(task, tag="jz seedance"):
    url = (task.get("content") or {}).get("video_url")
    if not url:
        raise RuntimeError(f"jz BytePlus Seedance: succeeded but no "
                           f"content.video_url: {json.dumps(task)[:300]}")
    usage = task.get("usage") or {}
    tokens = usage.get("total_tokens")
    print(f"[{tag}] done — {tokens} tokens "
          f"{format_cost(task.get('model', ''), tokens)}", flush=True)
    return (_download(url), task.get("id", ""), _applied(task),
            json.dumps(usage, ensure_ascii=False))


class jz_BytePlusSeedance:
    CATEGORY = "jz/api"
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "applied", "usage")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True,
                                      "default": "a cat walking through grass"}),
                "model": (model_ids("video", CURATED_VIDEO),
                          {"default": CURATED_VIDEO[0]}),
                "resolution": (["auto"] + RESOLUTIONS, {
                    "default": "720p",
                    "tooltip": "auto omits the flag and the api defaults to "
                               "1080p, the most expensive option"}),
                "ratio": (["auto"] + RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 0, "max": 60,
                                     "tooltip": "seconds; 0 omits the flag. "
                                                "cost scales linearly"}),
                "fps": ("INT", {"default": 24, "min": 0, "max": 60,
                                "tooltip": "0 omits the flag"}),
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1,
                                 "tooltip": "-1 omits it. deliberately NOT "
                                            "control_after_generate: a "
                                            "re-queue would spend again"}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                "reference_images": ("IMAGE", {
                    "tooltip": "every frame becomes a reference_image"}),
                "custom_model": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "region": (list(REGIONS), {"default": "ap-southeast"}),
                "poll_timeout": ("INT", {"default": 900, "min": 30, "max": 7200}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
            },
        }

    def generate(self, prompt, model, resolution, ratio, duration, fps,
                 camera_fixed, watermark, seed, first_frame=None,
                 last_frame=None, reference_images=None, custom_model="",
                 api_key="", region="ap-southeast", poll_timeout=900,
                 poll_interval=3):
        if not prompt.strip():
            raise ValueError("jz BytePlus Seedance: prompt is empty")
        model = resolve_model(model, custom_model)

        # validated before anything billable happens
        flags = build_flags(resolution, ratio, duration, fps,
                            camera_fixed, watermark, seed)

        content = [{"type": "text", "text": (prompt.strip() + " " + flags).strip()}]
        for tensor, role in ((first_frame, "first_frame"),
                             (last_frame, "last_frame")):
            if tensor is not None:
                content.append({"type": "image_url", "role": role,
                                "image_url": {"url": image_refs(tensor)[0]}})
        for uri in image_refs(reference_images):
            content.append({"type": "image_url", "role": "reference_image",
                            "image_url": {"url": uri}})

        tokens = estimate_tokens(resolution, ratio, duration, fps)
        est = (f"~{tokens:,} tokens {format_cost(model, tokens)}"
               if tokens else "cost unknown (auto resolution/ratio)")
        print(f"[jz seedance] submitting {model} {resolution} {ratio} "
              f"{duration}s @{fps} — {est}", flush=True)

        key = byteplus_key(api_key)
        task = api_post(region, TASKS, {"model": model, "content": content},
                        key, timeout=120)
        task_id = task.get("id")
        if not task_id:
            raise RuntimeError(f"jz BytePlus Seedance: no task id: "
                               f"{json.dumps(task)[:300]}")
        print(f"[jz seedance] task {task_id} — billed from here, cannot be "
              f"cancelled", flush=True)
        return _finish(_await(region, key, task_id, poll_timeout, poll_interval))


class jz_BytePlusSeedanceFetch:
    """Pick up a task by id — free, and the way to recover a paid job."""

    CATEGORY = "jz/api"
    RETURN_TYPES = ("VIDEO", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("video", "task_id", "applied", "usage")
    FUNCTION = "fetch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_id": ("STRING", {"default": "",
                                       "tooltip": "cgt-… from the generate "
                                                  "node; tasks live 48h"}),
                "wait": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "on: poll until it finishes. off: raise if it "
                               "is not done yet"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "region": (list(REGIONS), {"default": "ap-southeast"}),
                "poll_timeout": ("INT", {"default": 900, "min": 30, "max": 7200}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 30}),
            },
        }

    def fetch(self, task_id, wait, api_key="", region="ap-southeast",
              poll_timeout=900, poll_interval=3):
        task_id = task_id.strip()
        if not task_id:
            raise ValueError("jz BytePlus Seedance Fetch: task_id is empty")
        key = byteplus_key(api_key)
        if wait:
            task = _await(region, key, task_id, poll_timeout, poll_interval,
                          tag="jz seedance fetch")
        else:
            task = api_get(region, f"{TASKS}/{task_id}", key)
            if task.get("status") != "succeeded":
                raise RuntimeError(
                    f"jz BytePlus Seedance Fetch: task {task_id} is "
                    f"{task.get('status')}, not succeeded — turn `wait` on")
        return _finish(task, tag="jz seedance fetch")


NODE_CLASS_MAPPINGS = {"jz_BytePlusSeedance": jz_BytePlusSeedance,
                       "jz_BytePlusSeedanceFetch": jz_BytePlusSeedanceFetch}
NODE_DISPLAY_NAME_MAPPINGS = {
    "jz_BytePlusSeedance": "jz BytePlus Seedance (video)",
    "jz_BytePlusSeedanceFetch": "jz BytePlus Seedance Fetch (by task id)"}
