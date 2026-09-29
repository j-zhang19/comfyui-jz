"""Small helpers shared by node modules: wildcard sockets, list unwrapping,
and the separator table the text-splitting nodes share.
"""


class AnyType(str):
    """Equal to every type name — ComfyUI's wildcard socket convention.

    ComfyUI validates a link with `received_type != input_type`, so a type whose
    __ne__ is always False accepts anything. Used for the lazy if/else sockets.
    """

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class ComboAny(list):
    """Combo options that still accept a wired STRING.

    ComfyUI validates a link with `received_type != input_type` and gives up on
    a plain list (`if not isinstance(input_type, str): return False`), so a
    dropdown normally refuses every incoming wire — which would break saved
    workflows that feed jz Pad Calculator's aspect_ratio / resolution here.
    An always-equal __ne__ (the wildcard trick jz Switch / jz Fallback use)
    keeps those links valid, and serializes to /object_info exactly like a
    plain list so the frontend draws an ordinary dropdown.

    The cost: it accepts ANY type, so _pick() re-validates at run time.
    """

    def __ne__(self, other):
        return False

# how the list-taking nodes split their text input
SEPARATORS = {"newline": "\n", "comma": ",", "semicolon": ";", "pipe": "|"}


def scalar(v, default=None):
    """INPUT_IS_LIST hands every widget over as a 1-element list — unwrap it."""
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default


def format_usd(amount) -> str:
    """Cost as $0.0400, or $? when the provider reported none."""
    return f"${amount:.4f}" if isinstance(amount, (int, float)) else "$?"
