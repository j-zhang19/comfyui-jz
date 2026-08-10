"""jz Choice — pick from a list of choices by NAME, validated at run time.

The choices arrive as a STRING input (one per line, or another separator);
the pick is typed (or selected) in the `choice` widget. Unlike index-based
picking, reordering or extending the list upstream can never silently
change the selection — it either still matches or the node raises, listing
what was available.

web/jz_choice.js upgrades the `choice` widget to a dropdown when the
choices are wired from a node holding a literal string (a primitive /
string-literal node); with runtime-computed choices it stays a text field.
"""

_SEPARATORS = {"newline": "\n", "comma": ",", "semicolon": ";", "pipe": "|"}


class jz_Choice:
    CATEGORY = "jz/util"
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("value", "index")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "choices": ("STRING", {"forceInput": True, "multiline": True,
                                       "default": "",
                                       "tooltip": "the available options, one "
                                                  "per line (or per separator)"}),
                "choice": ("STRING", {"default": "",
                                      "tooltip": "the option to pick — must be "
                                                 "one of the choices"}),
            },
            "optional": {
                "separator": (list(_SEPARATORS), {"default": "newline"}),
            },
        }

    def pick(self, choices, choice, separator="newline"):
        entries = [e.strip() for e in choices.split(_SEPARATORS[separator])]
        entries = [e for e in entries if e]
        if not entries:
            raise ValueError("jz Choice: the choices list is empty")
        choice = choice.strip()
        if choice not in entries:
            raise ValueError(f"jz Choice: {choice!r} is not one of the "
                             f"choices: {', '.join(entries)}")
        return (choice, entries.index(choice))


NODE_CLASS_MAPPINGS = {"jz_Choice": jz_Choice}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_Choice": "jz Choice"}
