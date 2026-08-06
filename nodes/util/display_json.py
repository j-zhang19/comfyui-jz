"""jz Display JSON — pretty, collapsible JSON viewer.

Takes a STRING that should contain JSON and renders it in the node body as
a syntax-highlighted, collapsible tree (web/jz_display_json.js does the
rendering). Invalid JSON is shown raw with a parse-error banner instead of
failing the run — it's a display node, killing the workflow over it helps
nobody. Passes the prettified string through so it can be chained.
"""

import json


class jz_DisplayJSON:
    CATEGORY = "jz/util"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pretty",)
    FUNCTION = "show"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    def show(self, text):
        error = ""
        pretty = text if isinstance(text, str) else str(text)
        try:
            obj = json.loads(pretty)
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError) as e:
            error = f"not valid json: {e}"
        return {
            "ui": {"jz_json": [{"text": pretty, "error": error}]},
            "result": (pretty,),
        }


NODE_CLASS_MAPPINGS = {"jz_DisplayJSON": jz_DisplayJSON}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_DisplayJSON": "jz Display JSON"}
