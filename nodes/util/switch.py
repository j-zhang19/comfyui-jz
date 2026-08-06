"""jz Switch — boolean-driven lazy if/else.

`condition` picks which branch to output; the OTHER branch is never
executed (lazy). Both branches are optional: when the selected branch is
not connected, the node emits an ExecutionBlocker instead of a value, so
everything downstream is silently skipped ("if true output the image,
else output nothing").
"""

from comfy_execution.graph_utils import ExecutionBlocker


class _AnyType(str):
    """Equal to every type name — ComfyUI's wildcard socket convention."""

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class jz_Switch:
    CATEGORY = "jz/util"
    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "switch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_true": (_ANY, {"lazy": True}),
                "on_false": (_ANY, {"lazy": True}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def check_lazy_status(self, condition, on_true=None, on_false=None,
                          dynprompt=None, unique_id=None):
        # requesting an unconnected input is a hard NodeInputError, and an
        # unconnected socket is indistinguishable from an unevaluated lazy
        # one here (both None) — so check the graph for an actual link
        want = "on_true" if condition else "on_false"
        if dynprompt is not None and unique_id is not None:
            if want not in dynprompt.get_node(unique_id).get("inputs", {}):
                return []
        return [want]

    def switch(self, condition, on_true=None, on_false=None,
               dynprompt=None, unique_id=None):
        value = on_true if condition else on_false
        if value is None:
            # selected branch not connected -> mute everything downstream
            return (ExecutionBlocker(None),)
        return (value,)


NODE_CLASS_MAPPINGS = {"jz_Switch": jz_Switch}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_Switch": "jz Switch (lazy if/else)"}
