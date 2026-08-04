"""comfyui-jz — Jacques' personal node pack. Everything under the jz/ category.

Auto-discovers node modules: any nodes/**/*.py exporting NODE_CLASS_MAPPINGS
is merged in. Adding a node = dropping a file, no edits here.
"""
import importlib
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

_root = Path(__file__).parent
for _py in sorted((_root / "nodes").rglob("*.py")):
    if _py.name.startswith("_"):
        continue
    _rel = _py.relative_to(_root).with_suffix("")
    _mod = importlib.import_module("." + ".".join(_rel.parts), __package__)
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
