"""jz Composite Back — paste the original image over the generated one with a
feathered mask along the outpainted border (no API involved)."""
import numpy as np
import torch
from PIL import Image, ImageFilter


class jz_CompositeBack:
    """Composite original image back onto Gemini output with feathered edges."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "pad_left": ("INT", {"default": 0}),
                "pad_top": ("INT", {"default": 0}),
                "pad_right": ("INT", {"default": 0}),
                "pad_bottom": ("INT", {"default": 0}),
                "expand": ("INT", {"default": 4, "min": 0, "max": 64}),
                "max_filter_size": ("INT", {"default": 3, "min": 3, "max": 15, "step": 2}),
                "blur_radius": ("INT", {"default": 4, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "outpaint_mask", "paste_mask")
    FUNCTION = "composite"
    CATEGORY = "jz/image"

    def composite(
        self,
        generated_image: torch.Tensor,
        original_image: torch.Tensor,
        pad_left: int,
        pad_top: int,
        pad_right: int,
        pad_bottom: int,
        expand: int,
        max_filter_size: int,
        blur_radius: int,
    ):
        # Convert tensors to PIL
        gen_np = (generated_image[0].cpu().numpy() * 255).astype(np.uint8)
        gen_pil = Image.fromarray(gen_np)

        orig_np = (original_image[0].cpu().numpy() * 255).astype(np.uint8)
        orig_pil = Image.fromarray(orig_np)

        tw, th = gen_pil.size
        w, h = orig_pil.size

        # Build outpaint mask at padded dimensions (white=padded, black=original)
        outpaint_mask = Image.new("L", (tw, th), 255)
        outpaint_mask.paste(0, (pad_left, pad_top, pad_left + w, pad_top + h))

        # Dilate mask (grow padded area into original)
        # Ensure odd kernel size
        kernel = max_filter_size if max_filter_size % 2 == 1 else max_filter_size + 1
        for _ in range(expand):
            outpaint_mask = outpaint_mask.filter(ImageFilter.MaxFilter(kernel))

        # Blur for feathered edges
        if blur_radius > 0:
            outpaint_mask = outpaint_mask.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )

        # Invert: white=original area to keep, black=Gemini fills
        inverted_mask = Image.eval(outpaint_mask, lambda x: 255 - x)

        # Paste original onto Gemini result using feathered mask
        gen_pil.paste(
            orig_pil,
            (pad_left, pad_top),
            inverted_mask.crop((pad_left, pad_top, pad_left + w, pad_top + h)),
        )

        # Convert masks to tensors for debug output (grayscale -> RGB)
        outpaint_mask_np = np.array(outpaint_mask).astype(np.float32) / 255.0
        outpaint_mask_tensor = torch.from_numpy(outpaint_mask_np).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

        paste_mask_np = np.array(inverted_mask).astype(np.float32) / 255.0
        paste_mask_tensor = torch.from_numpy(paste_mask_np).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 3)

        # Convert back to tensor
        out_np = np.array(gen_pil).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).unsqueeze(0)

        return (out_tensor, outpaint_mask_tensor, paste_mask_tensor)


# key kept as "GeminiComposite" so saved workflows still resolve
NODE_CLASS_MAPPINGS = {"GeminiComposite": jz_CompositeBack}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiComposite": "jz Composite Back"}
