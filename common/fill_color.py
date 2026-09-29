"""Choosing a padding colour for an image.

Two strategies, both used by jz Pad Calculator:
  edge average — blend the border in, for padding that should disappear
  unique       — a colour provably absent from the image, so fill residue is
                 findable afterwards (jz Seam Repair looks for exactly this)
"""
import numpy as np
from PIL import Image
from scipy.spatial import KDTree


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


