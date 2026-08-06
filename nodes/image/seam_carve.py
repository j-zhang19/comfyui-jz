"""jz Seam Carve — content-aware resize by seam removal/insertion.

Implements Avidan & Shamir (2007) with the forward-energy criterion of
Rubinstein, Shamir & Avidan (2008), following the formulation in
arXiv:2608.04329 (Tosoni): L2 gradient energy on BT.601 luminance (or RGB),
central differences with periodic wrap, DP cumulative cost with backtracking,
sequential seam removal for reduction, ordered seam insertion with <=50%
multi-pass for enlargement, width processed before height.

Protection/removal use native ComfyUI MASK inputs instead of the paper's
color-coded mask image: protect_mask adds a large positive per-pixel weight
(seams avoid it), remove_mask a large negative one (seams eat it first).
To fully remove an object, reduce the width by at least the object's width,
then enlarge back.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # numpy fallback keeps the node functional without numba
    _HAS_NUMBA = False

_BIG = 1e6  # mask weight magnitude ("large" per the paper)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _backward_energy(rgb: np.ndarray, use_luma: bool) -> np.ndarray:
    """L2 gradient magnitude, central differences, periodic wrap."""
    if use_luma:
        chans = [_luma(rgb)]
    else:
        chans = [rgb[..., c] for c in range(rgb.shape[-1])]
    acc = np.zeros(rgb.shape[:2], dtype=np.float64)
    for ch in chans:
        dx = np.roll(ch, -1, axis=1) - np.roll(ch, 1, axis=1)
        dy = np.roll(ch, -1, axis=0) - np.roll(ch, 1, axis=0)
        acc += dx * dx + dy * dy
    return np.sqrt(acc)


def _dp_backward_np(e: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Cumulative-cost DP; returns the optimal vertical seam (col per row)."""
    n, m = e.shape
    cost = e + weight
    M = cost[0].copy()
    parent = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n):
        left = np.concatenate(([np.inf], M[:-1]))
        right = np.concatenate((M[1:], [np.inf]))
        stacked = np.stack([left, M, right])
        arg = np.argmin(stacked, axis=0)
        parent[i] = arg - 1
        M = cost[i] + stacked[arg, np.arange(m)]
    return _backtrack(M, parent)


def _dp_forward_np(gray: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Forward-energy DP (cost of edges CREATED by removal), periodic wrap."""
    n, m = gray.shape
    M = weight[0].astype(np.float64).copy()
    parent = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n):
        row, above = gray[i], gray[i - 1]
        cu = np.abs(np.roll(row, -1) - np.roll(row, 1))
        cl = cu + np.abs(above - np.roll(row, 1))
        cr = cu + np.abs(above - np.roll(row, -1))
        left = np.concatenate(([np.inf], M[:-1]))
        right = np.concatenate((M[1:], [np.inf]))
        stacked = np.stack([left + cl, M + cu, right + cr])
        arg = np.argmin(stacked, axis=0)
        parent[i] = arg - 1
        M = weight[i] + stacked[arg, np.arange(m)]
    return _backtrack(M, parent)


def _backtrack(last_row: np.ndarray, parent: np.ndarray) -> np.ndarray:
    n = parent.shape[0]
    seam = np.empty(n, dtype=np.int64)
    seam[-1] = int(np.argmin(last_row))
    for i in range(n - 1, 0, -1):
        seam[i - 1] = seam[i] + parent[i, seam[i]]
    return seam


if _HAS_NUMBA:

    @njit(cache=True, nogil=True)
    def _dp_backward_nb(cost):
        # tie-breaking mirrors the numpy path: left preferred, then up, right
        n, m = cost.shape
        parent = np.zeros((n, m), dtype=np.int8)
        prev = cost[0].copy()
        for i in range(1, n):
            cur = np.empty(m, dtype=np.float64)
            for j in range(m):
                if j > 0:
                    best = prev[j - 1]
                    arg = -1
                else:
                    best = np.inf
                    arg = -1
                if prev[j] < best:
                    best = prev[j]
                    arg = 0
                if j < m - 1 and prev[j + 1] < best:
                    best = prev[j + 1]
                    arg = 1
                parent[i, j] = arg
                cur[j] = cost[i, j] + best
            prev = cur
        seam = np.empty(n, dtype=np.int64)
        arg = 0
        best = prev[0]
        for j in range(1, m):
            if prev[j] < best:
                best = prev[j]
                arg = j
        seam[n - 1] = arg
        for i in range(n - 1, 0, -1):
            seam[i - 1] = seam[i] + parent[i, seam[i]]
        return seam

    @njit(cache=True, nogil=True)
    def _dp_forward_nb(gray, weight):
        n, m = gray.shape
        parent = np.zeros((n, m), dtype=np.int8)
        prev = weight[0].copy()
        for i in range(1, n):
            cur = np.empty(m, dtype=np.float64)
            for j in range(m):
                jl = j - 1 if j > 0 else m - 1  # periodic wrap (np.roll parity)
                jr = j + 1 if j < m - 1 else 0
                cu = abs(gray[i, jr] - gray[i, jl])
                if j > 0:
                    best = prev[j - 1] + cu + abs(gray[i - 1, j] - gray[i, jl])
                    arg = -1
                else:
                    best = np.inf
                    arg = -1
                cand = prev[j] + cu
                if cand < best:
                    best = cand
                    arg = 0
                if j < m - 1:
                    cand = prev[j + 1] + cu + abs(gray[i - 1, j] - gray[i, jr])
                    if cand < best:
                        best = cand
                        arg = 1
                parent[i, j] = arg
                cur[j] = weight[i, j] + best
            prev = cur
        seam = np.empty(n, dtype=np.int64)
        arg = 0
        best = prev[0]
        for j in range(1, m):
            if prev[j] < best:
                best = prev[j]
                arg = j
        seam[n - 1] = arg
        for i in range(n - 1, 0, -1):
            seam[i - 1] = seam[i] + parent[i, seam[i]]
        return seam


def _find_seam(rgb, weight, energy, use_luma):
    if energy == "forward":
        gray = _luma(rgb) if use_luma else rgb.mean(axis=-1)
        if _HAS_NUMBA:
            return _dp_forward_nb(np.ascontiguousarray(gray),
                                  np.ascontiguousarray(weight))
        return _dp_forward_np(gray, weight)
    e = _backward_energy(rgb, use_luma)
    if _HAS_NUMBA:
        return _dp_backward_nb(np.ascontiguousarray(e + weight))
    return _dp_backward_np(e, weight)


def _remove_seam(arr: np.ndarray, seam: np.ndarray) -> np.ndarray:
    n, m = arr.shape[:2]
    keep = np.ones((n, m), dtype=bool)
    keep[np.arange(n), seam] = False
    if arr.ndim == 3:
        return arr[keep].reshape(n, m - 1, arr.shape[2])
    return arr[keep].reshape(n, m - 1)


def _reduce_width(rgb, weight, k, energy, use_luma):
    for _ in range(k):
        seam = _find_seam(rgb, weight, energy, use_luma)
        rgb = _remove_seam(rgb, seam)
        weight = _remove_seam(weight, seam)
    return rgb, weight


def _enlarge_width(rgb, weight, k, energy, use_luma):
    """Ordered seam insertion; caller bounds k <= current_width // 2."""
    n, m = rgb.shape[:2]
    work, wwork = rgb.copy(), weight.copy()
    index = np.tile(np.arange(m), (n, 1))  # original column of each work pixel
    seams_orig = []
    for _ in range(k):
        seam = _find_seam(work, wwork, energy, use_luma)
        seams_orig.append(index[np.arange(n), seam])
        work = _remove_seam(work, seam)
        wwork = _remove_seam(wwork, seam)
        index = _remove_seam(index, seam)
    per_row = np.sort(np.stack(seams_orig, axis=1), axis=1)  # (n, k) columns

    out = np.empty((n, m + k, rgb.shape[2]), dtype=rgb.dtype)
    wout = np.empty((n, m + k), dtype=weight.dtype)
    for i in range(n):
        cols = per_row[i]
        row, wrow = rgb[i], weight[i]
        pieces, wpieces, prev = [], [], 0
        for c in cols:
            pieces.append(row[prev:c + 1])
            wpieces.append(wrow[prev:c + 1])
            neighbor = row[max(c - 1, 0)]
            pieces.append(((row[c] + neighbor) / 2.0)[None, :])
            wpieces.append(wrow[c:c + 1])
            prev = c + 1
        pieces.append(row[prev:])
        wpieces.append(wrow[prev:])
        out[i] = np.concatenate(pieces, axis=0)
        wout[i] = np.concatenate(wpieces, axis=0)
    return out, wout


def _resize_width(rgb, weight, target, energy, use_luma):
    m = rgb.shape[1]
    if target < m:
        return _reduce_width(rgb, weight, m - target, energy, use_luma)
    while rgb.shape[1] < target:
        # <=50% per pass: inserting seams comparable to the width degrades to
        # uniform scaling, so large enlargements run in several passes
        k = min(target - rgb.shape[1], rgb.shape[1] // 2)
        rgb, weight = _enlarge_width(rgb, weight, max(k, 1), energy, use_luma)
    return rgb, weight


def seam_carve(rgb, target_w, target_h, energy="forward", use_luma=True,
               weight=None):
    """rgb float [0,1] (H,W,C) -> content-aware resized copy."""
    n, m = rgb.shape[:2]
    weight = np.zeros((n, m), dtype=np.float64) if weight is None else weight
    target_w = m if target_w <= 0 else max(2, target_w)
    target_h = n if target_h <= 0 else max(2, target_h)
    if target_w != m:
        rgb, weight = _resize_width(rgb, weight, target_w, energy, use_luma)
    if target_h != n:  # height via transposition, same routine
        rgb = np.transpose(rgb, (1, 0, 2))
        weight = weight.T
        rgb, weight = _resize_width(rgb, weight, target_h, energy, use_luma)
        rgb = np.transpose(rgb, (1, 0, 2))
    return np.clip(rgb, 0.0, 1.0)


class jz_SeamCarve:
    CATEGORY = "jz/image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "carve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 16384,
                                         "tooltip": "0 = keep width; smaller "
                                                    "carves, larger inserts"}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 16384,
                                          "tooltip": "0 = keep height"}),
                "energy": (["forward", "backward"], {"default": "forward"}),
                "use_luma": ("BOOLEAN", {"default": True,
                                         "tooltip": "energy on BT.601 luma "
                                                    "(off = RGB gradients)"}),
            },
            "optional": {
                "protect_mask": ("MASK", {"tooltip": "seams avoid these pixels"}),
                "remove_mask": ("MASK", {"tooltip": "seams eat these pixels "
                                                    "first (carve at least the "
                                                    "object's width)"}),
            },
        }

    def carve(self, image, target_width, target_height, energy, use_luma,
              protect_mask=None, remove_mask=None):
        def _one(b: int) -> torch.Tensor:
            rgb = image[b].cpu().numpy().astype(np.float64)
            weight = np.zeros(rgb.shape[:2], dtype=np.float64)
            for mask, sign in ((protect_mask, +_BIG), (remove_mask, -_BIG)):
                if mask is None:
                    continue
                m_np = mask[min(b, mask.shape[0] - 1)].cpu().numpy()
                if m_np.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"jz Seam Carve: mask {m_np.shape} does not match "
                        f"image {rgb.shape[:2]}")
                weight += sign * m_np.astype(np.float64)
            out = seam_carve(rgb, target_width, target_height, energy,
                             use_luma, weight)
            return torch.from_numpy(out.astype(np.float32))

        n_frames = image.shape[0]
        if n_frames == 1:
            frames = [_one(0)]
        else:
            # frames are independent; numba kernels run with nogil so threads scale
            with ThreadPoolExecutor(max_workers=min(n_frames,
                                                    os.cpu_count() or 4)) as pool:
                frames = list(pool.map(_one, range(n_frames)))
        return (torch.stack(frames, dim=0),)


NODE_CLASS_MAPPINGS = {"jz_SeamCarve": jz_SeamCarve}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_SeamCarve": "jz Seam Carve"}
