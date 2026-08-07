"""jz Edge Sizes — width/height of an image sorted into long and short edge.

Saves the Get-Image-Size + math-expression dance when a node needs "the
larger dimension" (e.g. a resize target) regardless of orientation.
"""


class jz_EdgeSizes:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("long_edge", "short_edge")
    FUNCTION = "measure"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    def measure(self, image):
        # IMAGE is (batch, height, width, channels)
        h, w = int(image.shape[1]), int(image.shape[2])
        return (max(w, h), min(w, h))


NODE_CLASS_MAPPINGS = {"jz_EdgeSizes": jz_EdgeSizes}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_EdgeSizes": "jz Edge Sizes"}
