"""comfyui-jz — Jacques' personal node pack. One category: JZ/."""
from .gemini_outpaint import (
    NODE_CLASS_MAPPINGS as _GEMINI,
    NODE_DISPLAY_NAME_MAPPINGS as _GEMINI_NAMES,
)
from .nano_banana_pad import (
    NODE_CLASS_MAPPINGS as _PAD,
    NODE_DISPLAY_NAME_MAPPINGS as _PAD_NAMES,
)
from .outpaint_extras import (
    NODE_CLASS_MAPPINGS as _EXTRAS,
    NODE_DISPLAY_NAME_MAPPINGS as _EXTRAS_NAMES,
)

NODE_CLASS_MAPPINGS = {**_GEMINI, **_PAD, **_EXTRAS}
NODE_DISPLAY_NAME_MAPPINGS = {**_GEMINI_NAMES, **_PAD_NAMES, **_EXTRAS_NAMES}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
