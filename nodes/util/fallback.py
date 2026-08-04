"""jz Fallback — pass `primary` through if provided; otherwise evaluate and
return `fallback`.

The point is lazy evaluation: the `fallback` branch (e.g. an expensive
background generation) is NEVER EXECUTED when `primary` is connected and
produces a value. Typical wiring: primary = optional uploaded image,
fallback = the generation subgraph.

Works with any type (image, latent, string, ...) via the wildcard trick.
"""


class _AnyType(str):
    """Equal to every type name — ComfyUI's wildcard socket convention."""

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class jz_Fallback:
    CATEGORY = "jz/util"
    RETURN_TYPES = (_ANY, "BOOLEAN")
    RETURN_NAMES = ("value", "used_fallback")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "primary": (_ANY,),
                "fallback": (_ANY, {"lazy": True}),
            },
        }

    def check_lazy_status(self, primary=None, fallback=None):
        # only ask the executor to run the fallback branch when primary is absent
        if primary is None and fallback is None:
            return ["fallback"]
        return []

    def pick(self, primary=None, fallback=None):
        if primary is not None:
            return (primary, False)
        if fallback is None:
            raise ValueError("jz Fallback: neither primary nor fallback produced "
                             "a value — connect at least one input")
        return (fallback, True)


NODE_CLASS_MAPPINGS = {"jz_Fallback": jz_Fallback}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_Fallback": "jz Fallback (lazy if/else)"}
