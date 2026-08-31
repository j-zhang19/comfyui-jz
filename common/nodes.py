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

# how the list-taking nodes split their text input
SEPARATORS = {"newline": "\n", "comma": ",", "semicolon": ";", "pipe": "|"}


def scalar(v, default=None):
    """INPUT_IS_LIST hands every widget over as a 1-element list — unwrap it."""
    if isinstance(v, list):
        return v[0] if v else default
    return v if v is not None else default
