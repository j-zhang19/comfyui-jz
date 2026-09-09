"""jz Before/After Slider — animate a wipe between two images.

A divider sweeps across the frame revealing `after` over `before`, holds at the
far end, then sweeps back, so the batch loops seamlessly. Outputs the frames as
an IMAGE batch: pick the encoder yourself downstream (SaveAnimatedWEBP,
VHS_VideoCombine for a gif, SaveWEBM, CreateVideo).

Timing is two numbers. The hold at the starting end is split across the loop
seam — half at the head, half at the tail — so a looping player dwells there
exactly `hold_seconds`, the same as at the far end. sweep=1.5, hold=0.7, fps=20
gives 88 frames over 4.4s.

The handle is drawn once with PIL and alpha-composited per frame; the reveal
itself is tensor slicing, so source pixels never round-trip through 8-bit.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

import comfy.utils

from ...common.images import INTERPOLATION

ORIENTATIONS = ["vertical", "horizontal"]

EASINGS = {
    # smootherstep: zero 1st and 2nd derivative at both ends
    "smootherstep": lambda t: t * t * t * (t * (t * 6 - 15) + 10),
    "linear": lambda t: t,
}

# past this the output batch is big enough to be worth mentioning
_WARN_BYTES = 512 * 1024 ** 2


def positions(sweep: float, hold: float, fps: float, easing: str) -> list[float]:
    """Divider position per frame, 0 = far left/top, 1 = far right/bottom."""
    ease = EASINGS[easing]
    out = []
    for dur, a, b in ((hold / 2, 0.0, 0.0), (sweep, 0.0, 1.0), (hold, 1.0, 1.0),
                      (sweep, 1.0, 0.0), (hold / 2, 0.0, 0.0)):
        if dur <= 0:
            continue  # a segment that rounds to nothing contributes no frames
        n = max(1, round(dur * fps))
        out += [a + (b - a) * ease((i + 0.5) / n) for i in range(n)]
    return out


def build_handle(h: int, w: int) -> tuple[torch.Tensor, int]:
    """White divider line with a chevron grip, sized as a fraction of width."""
    r = max(3, round(w * 0.021))
    line_w = max(2, round(w * 0.004))
    hw = r * 2 + round(w * 0.03)
    lay = Image.new("RGBA", (hw, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cx, cy = hw // 2, h // 2
    d.rectangle([cx - line_w // 2 - 1, 0, cx + line_w // 2 + 1, h],
                fill=(0, 0, 0, 70))                      # shadow, then blurred
    lay = Image.alpha_composite(lay.filter(ImageFilter.GaussianBlur(3)), lay)
    d = ImageDraw.Draw(lay)
    d.rectangle([cx - line_w // 2, 0, cx + line_w // 2, h], fill=(255, 255, 255, 240))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, 245))
    for s in (-1, 1):
        ox = cx + s * round(r * 0.4)
        d.line([(ox - s * round(r * 0.18), cy - round(r * 0.24)), (ox, cy),
                (ox - s * round(r * 0.18), cy + round(r * 0.24))],
               fill=(30, 30, 30, 255), width=max(2, round(w * 0.0025)),
               joint="curve")
    return torch.from_numpy(np.array(lay).astype(np.float32) / 255.0), hw


class jz_BeforeAfterSlider:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("images", "frame_count", "fps")
    FUNCTION = "animate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "before": ("IMAGE",),
                "after": ("IMAGE", {"tooltip": "revealed by the wipe, on the "
                                               "leading side (left, or top when "
                                               "horizontal)"}),
                "scale": ("FLOAT", {
                    "default": 1.0, "min": 0.05, "max": 4.0, "step": 0.05,
                    "tooltip": "resize both images before animating — the main "
                               "lever on output size, and the handle scales "
                               "with it. 1.0 skips resampling entirely"}),
                "sweep_seconds": ("FLOAT", {
                    "default": 1.5, "min": 0.0, "max": 60.0, "step": 0.05,
                    "tooltip": "time for one crossing"}),
                "hold_seconds": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 60.0, "step": 0.05,
                    "tooltip": "dwell at each end; the start-end hold is split "
                               "across the loop seam"}),
                "fps": ("INT", {"default": 20, "min": 1, "max": 120}),
                "orientation": (ORIENTATIONS, {"default": "vertical"}),
                "easing": (list(EASINGS), {"default": "smootherstep"}),
                "handle": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "draw the divider line and chevron grip"}),
            },
            "optional": {
                "interpolation": (INTERPOLATION, {
                    "default": "lanczos",
                    "tooltip": "resample method for scale != 1.0"}),
            },
        }

    def animate(self, before, after, scale, sweep_seconds, hold_seconds, fps,
                orientation, easing, handle, interpolation="lanczos"):
        if before.shape[1:3] != after.shape[1:3]:
            raise ValueError(
                f"jz Before/After Slider: before is "
                f"{before.shape[2]}x{before.shape[1]} but after is "
                f"{after.shape[2]}x{after.shape[1]} — match them upstream "
                f"(jz Resize And Pad)")

        # a batch on either input animates its first frame only
        before, after = before[:1], after[:1]

        if scale != 1.0:
            h, w = int(before.shape[1]), int(before.shape[2])
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            before, after = (
                comfy.utils.common_upscale(t.permute(0, 3, 1, 2), nw, nh,
                                           interpolation, "disabled")
                .permute(0, 2, 3, 1) for t in (before, after))

        # horizontal is the vertical algorithm on a transposed frame
        flip = orientation == "horizontal"
        if flip:
            before, after = before.transpose(1, 2), after.transpose(1, 2)

        b = before[0]
        a = after[0]
        h, w, c = b.shape
        pos = positions(sweep_seconds, hold_seconds, fps, easing)
        if not pos:
            raise ValueError("jz Before/After Slider: sweep_seconds and "
                             "hold_seconds are both zero — nothing to animate")

        nbytes = len(pos) * h * w * c * 4
        if nbytes > _WARN_BYTES:
            print(f"[jz slider] {len(pos)} frames at {w}x{h} is "
                  f"{nbytes / 1024 ** 3:.2f} GB in the output batch — lower "
                  f"`scale` or the durations to shrink it", flush=True)

        grip, hw = build_handle(h, w) if handle else (None, 0)

        frames = []
        for p in pos:
            x = int(round(p * w))
            f = b.clone()
            if x > 0:
                f[:, :x] = a[:, :x]
            if grip is not None:
                # clip the grip to the frame; it hangs off both edges at 0 and 1
                x0 = x - hw // 2
                src = max(0, -x0)
                dst = max(0, x0)
                n = min(hw - src, w - dst)
                if n > 0:
                    band = grip[:, src:src + n]
                    alpha = band[..., 3:4]
                    f[:, dst:dst + n, :3] = (
                        alpha * band[..., :3]
                        + (1.0 - alpha) * f[:, dst:dst + n, :3])
            frames.append(f)

        out = torch.stack(frames, dim=0)
        if flip:
            out = out.transpose(1, 2)
        return (out.contiguous(), len(pos), float(fps))


NODE_CLASS_MAPPINGS = {"jz_BeforeAfterSlider": jz_BeforeAfterSlider}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_BeforeAfterSlider": "jz Before/After Slider"}
