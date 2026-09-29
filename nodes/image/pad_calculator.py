"""jz Pad Calculator — pad an image out to a supported Gemini output size.

The dimension table and the fitting logic live in common/gemini_dims.py; the
padding-colour search lives in common/fill_color.py. What is left here is the
padding geometry itself and the node.
"""
import numpy as np
import torch
from PIL import Image

from ...common.fill_color import find_unique_fill_color, get_edge_average_color
from ...common.gemini_dims import (DIMENSION_MAP, VALID_ASPECT_RATIOS,
                                   VALID_FIT_MODES, VALID_RESOLUTIONS,
                                   aspect_distance, find_best_fit,
                                   get_dimensions)


def calculate_padding(
    W: int,
    H: int,
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    mode: str = "superior",
) -> dict:
    """Calculate padding/cropping needed to reach target dimensions.

    Returns dict with pad values (positive = pad, negative = crop).
    """
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(
            f"Invalid aspect_ratio '{aspect_ratio}'. Valid: {VALID_ASPECT_RATIOS}"
        )
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(
            f"Invalid resolution '{resolution}'. Valid: {VALID_RESOLUTIONS}"
        )
    if mode not in VALID_FIT_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {VALID_FIT_MODES}")

    if aspect_ratio == "auto" and resolution == "auto":
        aspect_ratio, resolution = find_best_fit(W, H, mode=mode)
    elif aspect_ratio == "auto":
        img_area = W * H
        candidates = []
        for api_ratio, res_map in DIMENSION_MAP.items():
            if resolution in res_map:
                tw, th = res_map[resolution]
                if mode == "superior" and tw >= W and th >= H and (tw > W or th > H):
                    candidates.append((tw * th, api_ratio))
                elif mode == "inferior" and tw <= W and th <= H and (tw < W or th < H):
                    candidates.append(
                        (-tw * th, api_ratio)
                    )  # negative so sort picks largest
                elif mode == "closest":
                    # shape first, then size (see aspect_distance)
                    candidates.append(
                        ((aspect_distance(tw, th, W, H),
                          abs(tw * th - img_area)), api_ratio))
        if not candidates:
            raise ValueError(
                f"No matching dimension for image {W}x{H} at {resolution} in mode '{mode}'."
            )
        candidates.sort()
        aspect_ratio = candidates[0][1]
    elif resolution == "auto":
        if mode == "superior":
            for res in ["1K", "2K", "4K"]:
                tw, th = get_dimensions(aspect_ratio, res)
                if tw >= W and th >= H and (tw > W or th > H):
                    resolution = res
                    break
            else:
                raise ValueError(
                    f"Image {W}x{H} too large for {aspect_ratio} at any resolution."
                )
        elif mode == "inferior":
            for res in ["4K", "2K", "1K"]:
                tw, th = get_dimensions(aspect_ratio, res)
                if tw <= W and th <= H and (tw < W or th < H):
                    resolution = res
                    break
            else:
                raise ValueError(
                    f"Image {W}x{H} too small for {aspect_ratio} at any resolution."
                )
        elif mode == "closest":
            img_area = W * H
            best_res = None
            best_diff = float("inf")
            for res in ["1K", "2K", "4K"]:
                tw, th = get_dimensions(aspect_ratio, res)
                diff = abs(tw * th - img_area)
                if diff < best_diff:
                    best_diff = diff
                    best_res = res
            resolution = best_res

    tw, th = get_dimensions(aspect_ratio, resolution)

    # pad_h/pad_v: positive = pad, negative = crop
    pad_h = tw - W
    pad_v = th - H

    return {
        "pad_left": pad_h // 2,
        "pad_right": pad_h - pad_h // 2,
        "pad_top": pad_v // 2,
        "pad_bottom": pad_v - pad_v // 2,
        "target_width": tw,
        "target_height": th,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }


def pad_image(
    image: Image.Image,
    aspect_ratio: str = "auto",
    resolution: str = "auto",
    fill_color: tuple | None = None,
    mode: str = "superior",
) -> tuple[Image.Image, dict]:
    """Pad or crop an image to match Gemini API dimensions."""
    W, H = image.size
    padding_info = calculate_padding(W, H, aspect_ratio, resolution, mode=mode)

    tw = padding_info["target_width"]
    th = padding_info["target_height"]
    pl = padding_info["pad_left"]
    pt = padding_info["pad_top"]

    if image.mode != "RGB":
        image = image.convert("RGB")

    if tw >= W and th >= H:
        # Padding: target is larger or equal
        if fill_color is None:
            # fill_color = find_unique_fill_color(image)
            fill_color = get_edge_average_color(image)
        padding_info["fill_color"] = fill_color
        new_image = Image.new("RGB", (tw, th), fill_color)
        new_image.paste(image, (pl, pt))
    else:
        # Cropping: target is smaller, pad values are negative
        crop_left = -pl
        crop_top = -pt
        new_image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

    return new_image, padding_info


class jz_PadCalculator:
    """Compute padding and produce padded image for Gemini outpainting."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratio": (VALID_ASPECT_RATIOS, {"default": "auto"}),
                "resolution": (VALID_RESOLUTIONS, {"default": "auto"}),
                "mode": (VALID_FIT_MODES, {"default": "superior"}),
                # "unique": fill the padding with a color absent from the image
                # (magenta first) so fill-residue detection / seam repair can find
                # any leftover pad. "edge_average": blend the border in with the
                # mean edge color. Default keeps the historical edge_average
                # behaviour; the outpainting-v3 workflow selects "unique".
                "fill_mode": (["edge_average", "unique"], {"default": "edge_average"}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "IMAGE",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = (
        "pad_left",
        "pad_right",
        "pad_top",
        "pad_bottom",
        "target_w",
        "target_h",
        "aspect_ratio",
        "resolution",
        "fill_color",
        "padded_image",
        "outpaint_mask",
    )

    FUNCTION = "calculate"
    CATEGORY = "jz/image"

    def calculate(self, image, aspect_ratio: str, resolution: str, mode: str, fill_mode: str = "edge_average"):
        _, H, W, _ = image.shape

        # Convert tensor to PIL for pad_image
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # "unique" picks a color guaranteed absent from the image so leftover pad
        # can be detected/repaired downstream; "edge_average" blends the border.
        if fill_mode == "unique":
            fill_color = find_unique_fill_color(pil_img)
        else:
            fill_color = get_edge_average_color(pil_img)

        padded_pil, padding_info = pad_image(
            pil_img, aspect_ratio, resolution, fill_color=fill_color, mode=mode
        )

        tw = padding_info["target_width"]
        th = padding_info["target_height"]
        r, g, b = fill_color

        # Fill color image (BHWC, 0-1 float)
        fill = np.full(
            (1, th, tw, 3), [r / 255.0, g / 255.0, b / 255.0], dtype=np.float32
        )
        fill_tensor = torch.from_numpy(fill)

        # Padded image (BHWC, 0-1 float)
        padded_np = np.array(padded_pil).astype(np.float32) / 255.0
        padded_tensor = torch.from_numpy(padded_np).unsqueeze(0)

        # Outpaint mask: 1.0 = fill area, 0.0 = original content
        pl = padding_info["pad_left"]
        pt = padding_info["pad_top"]
        mask = np.zeros((th, tw), dtype=np.float32)
        mask[:pt, :] = 1.0  # top
        mask[pt + H :, :] = 1.0  # bottom
        mask[:, :pl] = 1.0  # left
        mask[:, pl + W :] = 1.0  # right
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return (
            padding_info["pad_left"],
            padding_info["pad_right"],
            padding_info["pad_top"],
            padding_info["pad_bottom"],
            tw,
            th,
            padding_info["aspect_ratio"],
            padding_info["resolution"],
            fill_tensor,
            padded_tensor,
            mask_tensor,
        )


# key kept as "GeminiPadCalculator" so saved workflows still resolve
NODE_CLASS_MAPPINGS = {"GeminiPadCalculator": jz_PadCalculator}

NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPadCalculator": "jz Pad Calculator"}
