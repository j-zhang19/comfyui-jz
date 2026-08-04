"""jz String Picker — pick one string from a list, randomly or by index.

- items: one entry per line (or choose another separator)
- mode "random": seeded pick — set the seed widget to "randomize" in ComfyUI
  to re-roll every queue, or fix it for reproducible picks
- mode "index": deterministic pick; the index wraps around (index % count),
  handy for sweeping through the list with an incrementing counter
Blank entries are skipped. Raises if the list is empty.
"""
import random

_SEPARATORS = {"newline": "\n", "comma": ",", "semicolon": ";", "pipe": "|"}


class jz_StringPicker:
    CATEGORY = "jz/util"
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("text", "picked_index", "count")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "items": ("STRING", {"multiline": True, "default": "",
                                     "tooltip": "one entry per line "
                                                "(or per separator)"}),
                "mode": (["random", "index"], {"default": "random"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                 "tooltip": "random mode: same seed = same pick"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                                  "tooltip": "index mode: wraps around the list"}),
            },
            "optional": {
                "separator": (list(_SEPARATORS), {"default": "newline"}),
            },
        }

    def pick(self, items, mode, seed, index, separator="newline"):
        entries = [e.strip() for e in items.split(_SEPARATORS[separator])]
        entries = [e for e in entries if e]
        if not entries:
            raise ValueError("jz String Picker: the items list is empty")
        if mode == "index":
            i = index % len(entries)
        else:
            i = random.Random(seed).randrange(len(entries))
        return (entries[i], i, len(entries))


NODE_CLASS_MAPPINGS = {"jz_StringPicker": jz_StringPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_StringPicker": "jz String Picker"}
