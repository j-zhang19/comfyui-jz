"""The Gemini / Nano-Banana output dimension table, and fitting an image to it.

Every supported aspect_ratio x resolution maps to one exact output size. Three
nodes need this — jz Pad Calculator, jz Gemini Generate and jz Resolution
Selector — so it lives here rather than being imported across nodes.

Pure data and geometry: no torch, no PIL, no I/O.
"""
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


