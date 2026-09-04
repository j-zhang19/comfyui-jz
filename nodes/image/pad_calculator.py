"""
ComfyUI custom node to compute padding for Gemini API target dimensions.
Maps standard API aspect ratios to actual output dimensions.
"""

import numpy as np
import torch
from PIL import Image
from scipy.spatial import KDTree

# Mapping: API aspect_ratio -> actual output (W, H) per resolution
DIMENSION_MAP = {
    "1:1": {
        "1K": (1024, 1024),
        "2K": (2048, 2048),
        "4K": (4096, 4096),
    },
    "5:4": {
        "1K": (1152, 928),
        "2K": (2304, 1856),
        "4K": (4608, 3712),
    },
    "4:5": {
        "1K": (928, 1152),
        "2K": (1856, 2304),
        "4K": (3712, 4608),
    },
    "4:3": {
        "1K": (1200, 896),
        "2K": (2400, 1792),
        "4K": (4800, 3584),
    },
    "3:4": {
        "1K": (896, 1200),
        "2K": (1792, 2400),
        "4K": (3584, 4800),
    },
    "3:2": {
        "1K": (1264, 848),
        "2K": (2528, 1696),
        "4K": (5056, 3392),
    },
    "2:3": {
        "1K": (848, 1264),
        "2K": (1696, 2528),
        "4K": (3392, 5056),
    },
    "16:9": {
        "1K": (1376, 768),
        "2K": (2752, 1536),
        "4K": (5504, 3072),
    },
    "9:16": {
        "1K": (768, 1376),
        "2K": (1536, 2752),
        "4K": (3072, 5504),
    },
    "21:9": {
        "1K": (1584, 672),
        "2K": (3168, 1344),
        "4K": (6336, 2688),
    },
}

VALID_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "5:4",
    "4:5",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "16:9",
    "9:16",
    "21:9",
]

VALID_RESOLUTIONS = ["auto", "1K", "2K", "4K"]

# Colors candidates for outpainting fill — checked in order, first absent color wins
_COLORS_CANDIDATES = [
    (255, 0, 255),  # magenta
    (0, 255, 0),  # green
    (0, 255, 255),  # cyan
    (255, 255, 0),  # yellow
    (0, 0, 0),  # black
    (255, 255, 255),  # white
    (255, 0, 0),  # red
    (0, 0, 255),  # blue
    (128, 0, 128),  # purple
    (0, 128, 128),  # teal
    (128, 128, 0),  # olive
    (1, 0, 0),  # near-black variants
    (0, 1, 0),
    (0, 0, 1),
    (254, 0, 255),
    (255, 0, 254),
]

def get_edge_average_color(image: Image.Image, band: int = 10) -> tuple[int, int, int]:
    """Compute mean RGB from the outermost pixel band of the image."""
    pixels = np.array(image.convert("RGB"))
    h, w = pixels.shape[:2]
    band = min(band, h // 2, w // 2)

    top = pixels[:band, :]
    bottom = pixels[-band:, :]
    left = pixels[band:-band, :band]
    right = pixels[band:-band, -band:]

    edge_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ]
    )

    mean = edge_pixels.mean(axis=0).astype(np.uint8)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def get_all_dimensions():
    """Flatten DIMENSION_MAP into list of (W, H, api_ratio, resolution)."""
    dims = []
    for api_ratio, res_map in DIMENSION_MAP.items():
        for res, (w, h) in res_map.items():
            dims.append((w, h, api_ratio, res))
    return dims


def get_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Get (W, H) for a specific aspect_ratio and resolution."""
    if aspect_ratio not in DIMENSION_MAP:
        raise ValueError(f"Unknown aspect_ratio: {aspect_ratio}")
    if resolution not in DIMENSION_MAP[aspect_ratio]:
        raise ValueError(f"Unknown resolution: {resolution}")
    return DIMENSION_MAP[aspect_ratio][resolution]


VALID_FIT_MODES = ["superior", "inferior", "closest"]


def aspect_distance(tw: int, th: int, W: int, H: int) -> float:
    """Shape mismatch between a candidate and the image; 1.0 means identical.

    Scale-free and symmetric. Area cannot stand in for this: w*h == h*w, so an
    area-only score cannot tell an aspect ratio from its own transpose, and
    "closest" answered 16:9 for every portrait 9:16 image.
    """
    r = (tw / max(th, 1)) / (W / max(H, 1))
    return max(r, 1.0 / r) if r > 0 else float("inf")


def find_best_fit(W: int, H: int, mode: str = "superior") -> tuple[str, str]:
    """Find best supported dimension for the image.

    Modes:
        superior: smallest dimension that fully contains the image (pad)
        inferior: largest dimension that fits inside the image (crop)
        closest: whichever dimension is closest in area
    """
    all_dims = get_all_dimensions()

    if mode == "superior":
        candidates = [
            (w, h, ar, res)
            for w, h, ar, res in all_dims
            if w >= W and h >= H and (w > W or h > H)
        ]
        if not candidates:
            raise ValueError(
                f"Image {W}x{H} exceeds all supported sizes. "
                f"Maximum supported is 6336x2688 (21:9 @ 4K) or 3072x5504 (9:16 @ 4K)."
            )
        candidates.sort(key=lambda x: x[0] * x[1])

    elif mode == "inferior":
        candidates = [
            (w, h, ar, res)
            for w, h, ar, res in all_dims
            if w <= W and h <= H and (w < W or h < H)
        ]
        if not candidates:
            raise ValueError(f"Image {W}x{H} is smaller than all supported sizes.")
        candidates.sort(key=lambda x: x[0] * x[1], reverse=True)

    elif mode == "closest":
        candidates = list(all_dims)
        if not candidates:
            raise ValueError("No supported dimensions available.")
        img_area = W * H
        # shape first, then size — area alone confuses a ratio with its transpose
        candidates.sort(key=lambda x: (aspect_distance(x[0], x[1], W, H),
                                       abs(x[0] * x[1] - img_area)))

    else:
        raise ValueError(f"Invalid mode '{mode}'. Valid: {VALID_FIT_MODES}")

    _, _, best_ar, best_res = candidates[0]
    return best_ar, best_res


def find_unique_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """Find an RGB color not present in the image."""
    pixels = np.array(image.convert("RGB")).reshape(-1, 3)
    # 24-bit occupancy table: one vectorised scatter, 16 MB flat. Boxing every
    # pixel into a python set costs ~2s and ~1 GB on a 4K image; this is ~0.03s.
    seen = np.zeros(1 << 24, dtype=bool)
    seen[pixels[:, 0].astype(np.uint32) << 16
         | pixels[:, 1].astype(np.uint32) << 8
         | pixels[:, 2].astype(np.uint32)] = True

    # Fast path: first candidate not in the image wins
    for r, g, b in _COLORS_CANDIDATES:
        if not seen[(r << 16) | (g << 8) | b]:
            return (r, g, b)

    # Slow path: all candidates present, find maximally distant color. Read the
    # distinct colors straight out of the table — np.unique(pixels, axis=0)
    # sorts all 16M rows and takes seconds for the same answer.
    packed = np.flatnonzero(seen).astype(np.uint32)
    unique = np.stack([packed >> 16, (packed >> 8) & 255, packed & 255],
                      axis=1).astype(np.float64)
    tree = KDTree(unique)

    best_color: tuple[int, int, int] = _COLORS_CANDIDATES[0]
    best_dist = 0.0

    for step, top_k in [(32, 8), (8, 3), (1, 1)]:
        axes = np.arange(0, 256, step)
        grid = (
            np.stack(np.meshgrid(axes, axes, axes), axis=-1)
            .reshape(-1, 3)
            .astype(np.float64)
        )
        dists, _ = tree.query(grid)
        top_indices = np.argpartition(dists, -top_k)[-top_k:]

        for idx in top_indices:
            if dists[idx] > best_dist:
                best_dist = dists[idx]
                best_color = (int(grid[idx][0]), int(grid[idx][1]), int(grid[idx][2]))

        if step == 1:
            break

        # Refine around top-k candidates
        next_candidates = []
        radius = step
        for idx in top_indices:
            center = grid[idx].astype(int)
            next_step = max(step // 4, 1)
            fine_axes = [
                np.arange(max(0, center[c] - radius),
                          min(256, center[c] + radius + 1), next_step)
                for c in range(3)
            ]
            fine = (
                np.stack(np.meshgrid(*fine_axes), axis=-1)
                .reshape(-1, 3)
                .astype(np.float64)
            )
            next_candidates.append(fine)

        refined = np.vstack(next_candidates)
        dists, _ = tree.query(refined)
        best_idx = np.argmax(dists)
        if dists[best_idx] > best_dist:
            best_dist = dists[best_idx]
            best_color = (
                int(refined[best_idx][0]),
                int(refined[best_idx][1]),
                int(refined[best_idx][2]),
            )

    return best_color


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
