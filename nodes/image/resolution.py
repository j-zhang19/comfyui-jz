"""jz Resolution Selector — aspect ratio to width/height, plain ratio names.

ComfyUI's core Resolution Selector labels its options "16:9 (Widescreen)" and
takes its input as a COMBO, which no STRING output can connect to
(validate_node_input rejects STRING against COMBO). That makes it impossible to
drive from jz Pad Calculator's resolved aspect_ratio output.

This is the same idea with the parenthesised names dropped, the ratio list taken
from DIMENSION_MAP (the 10 pad calculator knows — a superset of core's 8, adding
4:5 and 5:4), and an aspect_ratio_in STRING socket you can actually wire.

Two ways to size the result:
- table:      the exact dimensions Gemini emits, straight out of DIMENSION_MAP
- megapixels: core Resolution Selector's formula, for any ratio and any target

They disagree slightly by design — 16:9 at 1 MP is 1360x768 by the formula but
1376x768 in the table. Use table when the number has to match the API.
"""
import math

from .pad_calculator import DIMENSION_MAP

ASPECT_RATIOS = list(DIMENSION_MAP)
RESOLUTIONS = list(next(iter(DIMENSION_MAP.values())))
MODES = ["table", "megapixels"]


def parse_ratio(text: str) -> tuple[int, int]:
    """'16:9' -> (16, 9). Also accepts core's '16:9 (Widescreen)' form."""
    s = (text or "").strip().split()[0] if (text or "").strip() else ""
    w, _, h = s.partition(":")
    try:
        wi, hi = int(w), int(h)
    except ValueError:
        raise ValueError(
            f"jz Resolution Selector: aspect_ratio {text!r} is not a w:h ratio "
            f"— use one of {', '.join(ASPECT_RATIOS)}, or any 'w:h'") from None
    if wi <= 0 or hi <= 0:
        raise ValueError(f"jz Resolution Selector: aspect_ratio {text!r} has a "
                         f"zero or negative side")
    return wi, hi


class jz_ResolutionSelector:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {
                    "default": "1:1",
                    "tooltip": "plain ratios, no parenthesised names — wire "
                               "jz Pad Calculator's aspect_ratio into "
                               "aspect_ratio_in to drive this from a graph"}),
                "mode": (MODES, {
                    "default": "table",
                    "tooltip": "table = the exact dimensions Gemini emits; "
                               "megapixels = core Resolution Selector's formula"}),
                "resolution": (RESOLUTIONS, {
                    "default": "1K", "tooltip": "table mode only"}),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "megapixels mode only — 1.0 MP is about "
                               "1024x1024 square"}),
                "multiple": ("INT", {
                    "default": 8, "min": 8, "max": 128, "step": 4,
                    "tooltip": "megapixels mode only — each side is rounded to "
                               "a multiple of this"}),
            },
            "optional": {
                "aspect_ratio_in": ("STRING", {
                    "forceInput": True, "default": "",
                    "tooltip": "overrides the dropdown when connected — e.g. "
                               "jz Pad Calculator's aspect_ratio output"}),
            },
        }

    def select(self, aspect_ratio, mode, resolution, megapixels, multiple,
               aspect_ratio_in=""):
        # a connected string beats the dropdown
        wired = aspect_ratio_in.strip() if isinstance(aspect_ratio_in, str) else ""
        ratio = wired or aspect_ratio

        if mode == "table":
            # normalise so core's "16:9 (Widescreen)" also finds the table entry
            key = f"{'%d:%d' % parse_ratio(ratio)}"
            if key not in DIMENSION_MAP:
                raise ValueError(
                    f"jz Resolution Selector: no table entry for {key!r} — "
                    f"table mode covers {', '.join(ASPECT_RATIOS)}; switch to "
                    f"megapixels mode for any other ratio")
            w, h = DIMENSION_MAP[key][resolution]
            return (int(w), int(h))

        w_ratio, h_ratio = parse_ratio(ratio)
        scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
        return (round(w_ratio * scale / multiple) * multiple,
                round(h_ratio * scale / multiple) * multiple)


NODE_CLASS_MAPPINGS = {"jz_ResolutionSelector": jz_ResolutionSelector}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ResolutionSelector": "jz Resolution Selector"}
