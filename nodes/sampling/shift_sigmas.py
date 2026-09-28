"""jz Shift Sigmas — resolution-dependent flow-match shift applied to SIGMAS.

Flow-match models re-time their schedule by image size: the more tokens, the
more the schedule is pushed toward high noise.

    mu    = base_shift + (max_shift - base_shift) * (tokens - min_tok)
                                                  / (max_tok - min_tok)
    sigma = e^mu / (e^mu + 1/t - 1)

ComfyUI has both halves of this but never the combination. ManualSigmas emits
explicit sigmas with no shift; ModelSamplingFlux computes exactly this mu (and
the identical token count, width*height/(8*8*2*2) == (W/16)*(H/16)) but patches
the MODEL rather than producing SIGMAS. So there is no way to take a schedule
and shift it. This node is that missing step.

One node covers the different model families, which differ only in constants:
    Flux          base 0.5  max 1.15  tokens 256..4096
    Qwen-Image    base 0.5  max 0.90  tokens 256..8192

Feed it RAW, UNSHIFTED values in (0, 1] — a schedule that has already been
shifted (e.g. from BasicScheduler on a flow-match model) would be shifted twice.
"""
import math

import torch


class jz_ShiftSigmas:
    CATEGORY = "jz/sampling"
    RETURN_TYPES = ("SIGMAS", "FLOAT", "INT")
    RETURN_NAMES = ("sigmas", "mu", "tokens")
    FUNCTION = "shift"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {
                    "tooltip": "raw unshifted values in (0,1] — pair with "
                               "ManualSigmas, e.g. 1.0, 0.9375, 0.875, 0.75, "
                               "0.5, 0.25 for qwen viggle-turbo"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384,
                                  "step": 8,
                                  "tooltip": "image pixels, ignored when a "
                                             "latent is connected"}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384,
                                   "step": 8}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0,
                                         "max": 100.0, "step": 0.01,
                                         "tooltip": "mu at min_tokens"}),
                "max_shift": ("FLOAT", {"default": 0.9, "min": 0.0,
                                        "max": 100.0, "step": 0.01,
                                        "tooltip": "mu at max_tokens. "
                                                   "qwen-image 0.9, flux 1.15"}),
                "min_tokens": ("INT", {"default": 256, "min": 1, "max": 1 << 20}),
                "max_tokens": ("INT", {"default": 8192, "min": 1, "max": 1 << 20,
                                       "tooltip": "qwen-image 8192, flux 4096"}),
                "append_zero": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "samplers need a terminal 0 and ManualSigmas "
                               "does not add one; skipped if already present"}),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "take the resolution from the latent being "
                               "sampled instead of the width/height widgets"}),
            },
        }

    @staticmethod
    def _tokens(width, height, latent):
        """Token count on the 1/16 grid — the same measure core's flux shift uses."""
        if latent is None:
            return max(1, round(width / 16) * round(height / 16))
        s = latent["samples"]
        # a latent is stored at its own downscale ratio; rescale to the /16 grid.
        # the key is optional, and assuming /16 when it is absent is how the
        # original overcounts a plain /8 EmptyLatentImage by 4x
        ratio = latent.get("downscale_ratio_spacial") or 16
        r = ratio / 16
        return max(1, round(s.shape[-2] * r) * round(s.shape[-1] * r))

    def shift(self, sigmas, width, height, base_shift, max_shift,
              min_tokens, max_tokens, append_zero, latent=None):
        if sigmas is None or sigmas.numel() == 0:
            raise ValueError("jz Shift Sigmas: the sigmas input is empty")

        tokens = self._tokens(width, height, latent)
        span = max_tokens - min_tokens
        # a zero span means one fixed mu rather than a division by zero
        mu = base_shift if span == 0 else (
            base_shift + (max_shift - base_shift) * (tokens - min_tokens) / span)

        t = sigmas.detach().to(torch.float64).flatten()
        out = torch.zeros_like(t)
        # 1/t is inf at t=0; it happens to yield 0, but skip it rather than
        # lean on inf arithmetic
        nz = t != 0
        e = math.exp(mu)
        out[nz] = e / (e + (1.0 / t[nz] - 1.0))

        if append_zero and float(out[-1]) != 0.0:
            out = torch.cat([out, out.new_zeros(1)])
        return (out.float(), float(mu), int(tokens))


NODE_CLASS_MAPPINGS = {"jz_ShiftSigmas": jz_ShiftSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"jz_ShiftSigmas": "jz Shift Sigmas (flow match)"}
