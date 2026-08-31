"""jz Fallback — pass `primary` through if provided; otherwise evaluate and
return `fallback`.

The point is lazy evaluation: the `fallback` branch (e.g. an expensive
background generation) is NEVER EXECUTED when `primary` is connected and
produces a value. Typical wiring: primary = optional uploaded image,
fallback = the generation subgraph.

Works with any type (image, latent, string, ...) via the wildcard trick.
"""

from ...common.nodes import ANY


class jz_Fallback:
    CATEGORY = "jz/util"
    RETURN_TYPES = (ANY, "BOOLEAN")
    RETURN_NAMES = ("value", "used_fallback")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "primary": (ANY,),
                "fallback": (ANY, {"lazy": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def check_lazy_status(self, primary=None, fallback=None,
                          dynprompt=None, unique_id=None):
        # only ask the executor to run the fallback branch when primary is
        # absent — and only if fallback is actually connected (requesting an
        # unconnected input is a hard NodeInputError)
        if primary is None and fallback is None:
            if dynprompt is not None and unique_id is not None:
                if "fallback" not in dynprompt.get_node(unique_id).get("inputs", {}):
                    return []
            return ["fallback"]
        return []

    def pick(self, primary=None, fallback=None,
             dynprompt=None, unique_id=None):
        if primary is not None:
            return (primary, False)
        if fallback is None:
            raise ValueError("jz Fallback: neither primary nor fallback produced "
                             "a value — connect at least one input")
        return (fallback, True)


NODE_CLASS_MAPPINGS = {"jz_Fallback": jz_Fallback}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_Fallback": "jz Fallback (lazy if/else)"}
