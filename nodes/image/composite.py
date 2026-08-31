"""jz Composite — paste a source image onto a destination at a named anchor.

Replaces the Get-Image-Size + (a-b)/2 math cluster for the common "put this
image in the center / corner of that one" case. Placement = anchor + x/y
offset, with a margin that pads corner/edge anchors away from the borders.

- optional MASK input: source is blended through it; without a mask, a
  source with an alpha channel blends through its own alpha, otherwise
  it's a plain rectangle paste
- no scaling here (resize upstream); raises if the source doesn't fully
  fit at the computed position
"""

ANCHORS = [
    "center",
    "top-left", "top-center", "top-right",
    "middle-left", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


class jz_Composite:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "anchor": (ANCHORS, {"default": "center"}),
                "offset_x": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "offset_y": ("INT", {"default": 0, "min": -16384, "max": 16384}),
                "margin": ("INT", {"default": 0, "min": 0, "max": 16384,
                                   "tooltip": "padding from the borders for "
                                              "corner/edge anchors"}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    @staticmethod
    def _axis_pos(anchor_part: str, dest_size: int, src_size: int, margin: int) -> int:
        if anchor_part == "start":
            return margin
        if anchor_part == "end":
            return dest_size - src_size - margin
        return (dest_size - src_size) // 2  # center: margin does not apply

    def composite(self, destination, source, anchor, offset_x, offset_y,
                  margin, mask=None):
        dh, dw = destination.shape[1], destination.shape[2]
        sh, sw = source.shape[1], source.shape[2]

        vert, _, horiz = anchor.partition("-") if "-" in anchor else ("center", "", "center")
        x = self._axis_pos({"left": "start", "right": "end"}.get(horiz, "center"),
                           dw, sw, margin) + offset_x
        y = self._axis_pos({"top": "start", "bottom": "end"}.get(vert, "center"),
                           dh, sh, margin) + offset_y

        if x < 0 or y < 0 or x + sw > dw or y + sh > dh:
            raise ValueError(
                f"jz Composite: source {sw}x{sh} at ({x},{y}) does not fit in "
                f"destination {dw}x{dh} (anchor={anchor}, offset=({offset_x},"
                f"{offset_y}), margin={margin}) — resize the source or adjust "
                f"the placement")

        # broadcast batches: 1-frame source/mask repeats over the destination batch
        batch = destination.shape[0]
        if source.shape[0] not in (1, batch):
            raise ValueError(f"jz Composite: source batch {source.shape[0]} "
                             f"does not match destination batch {batch}")
        if source.shape[0] == 1 and batch > 1:
            source = source.expand(batch, -1, -1, -1)

        # blend weights: explicit mask > source alpha > opaque rectangle
        src_rgb = source[..., :3]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[-2:] != (sh, sw):
                raise ValueError(f"jz Composite: mask {mask.shape[-1]}x"
                                 f"{mask.shape[-2]} does not match source {sw}x{sh}")
            if mask.shape[0] not in (1, batch):
                raise ValueError(f"jz Composite: mask batch {mask.shape[0]} "
                                 f"does not match destination batch {batch}")
            if mask.shape[0] == 1 and batch > 1:
                mask = mask.expand(batch, -1, -1)
            alpha = mask.unsqueeze(-1)
        elif source.shape[-1] == 4:
            alpha = source[..., 3:4]
        else:
            alpha = None

        out = destination[..., :3].clone()
        region = out[:, y:y + sh, x:x + sw, :]
        out[:, y:y + sh, x:x + sw, :] = (
            src_rgb if alpha is None else alpha * src_rgb + (1.0 - alpha) * region
        )
        return (out,)


NODE_CLASS_MAPPINGS = {"jz_Composite": jz_Composite}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_Composite": "jz Composite"}
