"""jz Image Sanity — flag degenerate frames before they poison the workflow.

An image API occasionally hands back a flat black frame, a blown-out white
one, or the pad colour echoed over the whole canvas. Downstream nodes process
it perfectly happily and the failure only surfaces in the saved file. This
node measures every frame and REPORTS — it never fails the run over a bad
frame, so you branch on `ok` with jz Switch / jz Fallback and decide for
yourself what a failure means (retry, substitute, skip).

Checks: empty (0-px, always on), flat, too dark, too bright, fully
transparent — the last four toggleable with their own thresholds.

Flatness is measured PER CHANNEL. A pure red frame has channel stds of 0 but
a std over the whole tensor of ~0.47, so one global std would call it
textured — exactly the frame you wanted caught. max(per-channel std) is the
honest measure, and it's what the `std` output reports.

Per-frame outputs are LISTs (a batch of 4 gives 4 verdicts); `report` (JSON,
for jz Display JSON) and `all_ok` are scalars for whole-batch decisions.
"""

import json

from ...common.images import luma as _luma
from ...common.nodes import scalar

# below this an alpha channel counts as fully transparent (float noise guard)
_ALPHA_EPS = 1e-6


def _frames(v):
    """Flatten batches/lists into per-frame tensors: IMAGE -> (H,W,C), MASK -> (H,W)."""
    out = []
    for t in v if isinstance(v, list) else [v]:
        if t is None:
            continue
        if t.dim() == 2:  # a bare (H,W) mask
            out.append(t)
        else:
            out.extend(t)
    return out


def _hex(rgb_mean) -> str:
    return "#%02x%02x%02x" % tuple(
        max(0, min(255, int(round(float(c) * 255.0)))) for c in rgb_mean
    )


class jz_ImageSanity:
    CATEGORY = "jz/image"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING", "FLOAT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("image", "ok", "reason", "std", "mean", "report", "all_ok")
    # per-frame slots are lists; report/all_ok are single values describing the
    # whole batch. a True slot gets extend()ed — marking `report` as a list
    # would splatter the json string one character per output.
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False)
    FUNCTION = "check"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "check_flat": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "fail frames with no variation — flat black, "
                        "flat white, any single fill colour",
                    },
                ),
                "min_std": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "flat when the largest per-channel std falls "
                        "below this; watch the std output to tune it",
                    },
                ),
                "check_dark": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "fail near-black frames that carry just enough "
                        "noise to pass the flat check",
                    },
                ),
                "min_mean": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "too dark when mean luma is below this",
                    },
                ),
                "check_bright": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "fail blown-out near-white frames"},
                ),
                "max_mean": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "too bright when mean luma is above this",
                    },
                ),
                "check_transparent": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "fail frames whose alpha is zero everywhere — "
                        "nothing visible, however textured the rgb is",
                    },
                ),
            },
            "optional": {
                "alpha": (
                    "MASK",
                    {
                        "tooltip": "transparency for a 3-channel image; "
                        "LoadImage's MASK output goes here"
                    },
                ),
                "invert_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "LoadImage masks are 1 = transparent; keep on "
                        "when wiring that, turn off if your mask is "
                        "1 = opaque",
                    },
                ),
            },
        }

    @staticmethod
    def _measure(frame, mask, invert_alpha, index, total, cfg):
        """One frame -> (ok, reason, std, mean, stats dict)."""
        h, w = int(frame.shape[0]), int(frame.shape[1])
        stats = {"frame": index + 1, "width": w, "height": h}

        if h == 0 or w == 0:
            stats.update(
                std=0.0, mean=0.0, coverage=0.0, ok=False, reason=f"empty ({w}x{h})"
            )
            return False, stats["reason"], 0.0, 0.0, stats

        rgb = frame[..., :3].float()
        # correction=0: population std, so a 1-px frame gives 0 rather than nan
        std = float(rgb.std(dim=(0, 1), correction=0).max())
        luma = _luma(rgb)
        mean = float(luma.mean())

        # opacity: the 4th image channel and/or the wired mask
        opacity = frame[..., 3] if frame.shape[-1] == 4 else None
        if mask is not None:
            # LoadImage emits a constant 64x64 placeholder mask when the image
            # has no alpha channel — treat that as "no mask wired"
            placeholder = (
                tuple(mask.shape[-2:]) == (64, 64)
                and (h, w) != (64, 64)
                and bool((mask == mask.flatten()[0]).all())
            )
            if not placeholder:
                if tuple(mask.shape[-2:]) != (h, w):
                    raise ValueError(
                        f"jz Image Sanity: alpha {mask.shape[-1]}x{mask.shape[-2]} "
                        f"does not match image {w}x{h} on frame {index + 1}"
                    )
                m = 1.0 - mask if invert_alpha else mask
                opacity = m if opacity is None else opacity * m

        coverage = 1.0 if opacity is None else float(opacity.mean())
        stats.update(
            std=round(std, 6), mean=round(mean, 6), coverage=round(coverage, 6)
        )

        fails = []
        if cfg["flat"] and std < cfg["min_std"]:
            fails.append(
                f"flat {_hex(rgb.mean(dim=(0, 1)))} "
                f"(std {std:.4f} < {cfg['min_std']:.4f})"
            )
        if cfg["dark"] and mean < cfg["min_mean"]:
            fails.append(f"too dark (mean {mean:.4f} < {cfg['min_mean']:.4f})")
        if cfg["bright"] and mean > cfg["max_mean"]:
            fails.append(f"too bright (mean {mean:.4f} > {cfg['max_mean']:.4f})")
        if (
            cfg["transparent"]
            and opacity is not None
            and float(opacity.max()) <= _ALPHA_EPS
        ):
            fails.append("fully transparent (alpha max 0.0000)")

        ok = not fails
        # the frame prefix only earns its place in a batch
        prefix = "" if total == 1 else f"frame {index + 1}/{total}: "
        reason = "" if ok else prefix + "; ".join(fails)
        stats.update(ok=ok, reason=reason)
        return ok, reason, std, mean, stats

    def check(
        self,
        image,
        check_flat,
        min_std,
        check_dark,
        min_mean,
        check_bright,
        max_mean,
        check_transparent,
        alpha=None,
        invert_alpha=True,
    ):
        settings = {
            "flat": bool(scalar(check_flat, True)),
            "min_std": float(scalar(min_std, 0.01)),
            "dark": bool(scalar(check_dark, True)),
            "min_mean": float(scalar(min_mean, 0.02)),
            "bright": bool(scalar(check_bright, True)),
            "max_mean": float(scalar(max_mean, 0.98)),
            "transparent": bool(scalar(check_transparent, True)),
        }
        invert = bool(scalar(invert_alpha, True))

        frames = _frames(image)
        if not frames:
            raise ValueError("jz Image Sanity: no images provided")
        masks = _frames(alpha) if alpha is not None else []
        if len(masks) == 1 and len(frames) > 1:
            masks = masks * len(frames)  # one mask broadcasts over the batch
        if masks and len(masks) != len(frames):
            raise ValueError(
                f"jz Image Sanity: {len(masks)} alpha masks for "
                f"{len(frames)} frames — wire one per frame or one "
                f"for all"
            )

        images, oks, reasons, stds, means, per_frame = [], [], [], [], [], []
        for i, frame in enumerate(frames):
            ok, reason, std, mean, stats = self._measure(
                frame, masks[i] if masks else None, invert, i, len(frames), settings
            )
            images.append(frame.unsqueeze(0))  # pass through untouched
            oks.append(ok)
            reasons.append(reason)
            stds.append(std)
            means.append(mean)
            per_frame.append(stats)

        all_ok = all(oks)
        report = json.dumps(
            {
                "frames": len(frames),
                "ok": all_ok,
                "failed": [i + 1 for i, o in enumerate(oks) if not o],
                "settings": settings,
                "per_frame": per_frame,
            },
            ensure_ascii=False,
        )
        return (images, oks, reasons, stds, means, report, all_ok)


NODE_CLASS_MAPPINGS = {"jz_ImageSanity": jz_ImageSanity}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ImageSanity": "jz Image Sanity"}
