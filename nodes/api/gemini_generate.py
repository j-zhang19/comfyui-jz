import torch
import numpy as np
import requests
import base64
import json
import os
import subprocess
import tempfile
import time
import re
from io import BytesIO
from PIL import Image



def _get_access_token(
    service_account_b64: str,
    scope: str = "https://www.googleapis.com/auth/generative-language",
) -> tuple[str, str]:
    """Generate OAuth2 access token from base64-encoded service account JSON.

    Args:
        service_account_b64: base64-encoded service account JSON.
        scope: OAuth scope to request. Use the generative-language scope for the
            AI Studio (generativelanguage.googleapis.com) endpoint, or the
            cloud-platform scope for Vertex AI (aiplatform.googleapis.com).

    Returns:
        Tuple of (access_token, project_id)
    """
    sa_json = base64.b64decode(service_account_b64).decode("utf-8")
    sa_data = json.loads(sa_json)

    client_email = sa_data["client_email"]
    private_key = sa_data["private_key"]
    project_id = sa_data["project_id"]

    header = {"alg": "RS256", "typ": "JWT"}
    now = int(time.time())

    payload = {
        "iss": client_email,
        "scope": scope,
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now,
        "exp": now + 3600,
    }

    header_b64 = (
        base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )

    message = f"{header_b64}.{payload_b64}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pem", delete=False) as tmp:
        tmp.write(private_key)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", tmp_path, "-binary"],
            input=message.encode(),
            capture_output=True,
            check=True,
        )
        signature = base64.urlsafe_b64encode(proc.stdout).decode().rstrip("=")
    finally:
        os.unlink(tmp_path)

    jwt_token = f"{header_b64}.{payload_b64}.{signature}"

    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token,
        },
    )
    resp.raise_for_status()
    access_token = resp.json()["access_token"]

    return access_token, project_id


# generativelanguage.googleapis.com (AI Studio) only accepts these HarmCategory
# enum values. The IMAGE_* and JAILBREAK categories are Vertex-only and cause a
# 400 ("Invalid value at 'safety_settings[..].category'") on this endpoint.
GLA_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
]

# Vertex AI (aiplatform.googleapis.com) additionally supports image-specific
# categories and a jailbreak category.
VERTEX_SAFETY_SETTINGS = GLA_SAFETY_SETTINGS + [
    {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_JAILBREAK", "threshold": "OFF"},
]

GLA_SCOPE = "https://www.googleapis.com/auth/generative-language"
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class jz_GeminiGenerate:
    """ComfyUI node for Gemini image generation via Vertex AI."""

    MODELS = [
        # Vertex AI (aiplatform) publisher model names — used with backend="vertex".
        "gemini-3-pro-image",
        "gemini-3.1-flash-image",
        # AI Studio (generativelanguage) names — used with backend="generativelanguage".
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-image-preview",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate an image of a cute dog.",
                    },
                ),
                "service_account_base64": ("STRING", {"default": ""}),
                "model": (cls.MODELS + ["custom"], {"default": cls.MODELS[0]}),
                "location": ("STRING", {"default": "us-central1"}),
                "aspect_ratio": ("STRING", {"default": "1:1"}),
                "resolution": ("STRING", {"default": "1K"}),
                # Cache-buster only: NOT sent to Gemini (the REST API has no seed).
                # ComfyUI caches a node whose inputs are unchanged, so with fixed
                # params it would never re-call the API. A changing seed changes the
                # cache key, forcing a fresh (stochastic) generation each run.
                # Set the control to "fixed" to reuse the cached result instead.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "backend": (["generativelanguage", "vertex"], {"default": "generativelanguage"}),
                "custom_model": ("STRING", {"default": ""}),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_6": ("IMAGE",),
                "image_7": ("IMAGE",),
                "image_8": ("IMAGE",),
                "image_9": ("IMAGE",),
                "image_10": ("IMAGE",),
                "images": ("IMAGE", {"tooltip": "a LIST of images (e.g. from "
                                                "jz Resize Long Edge) — all "
                                                "frames are sent"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8,
                                       "tooltip": "N parallel API calls -> N "
                                                  "images out (one shared "
                                                  "token, per-call retries)"}),
            },
        }

    # outputs appended (never reordered) so existing links keep their slots
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("image", "usage", "total_tokens")
    FUNCTION = "generate"
    CATEGORY = "jz/api"
    INPUT_IS_LIST = True

    def _tensor_to_base64(self, image: torch.Tensor) -> str:
        """Convert a ComfyUI image tensor (BHWC, 0-1 float) to a base64 PNG string."""
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def generate(
        self,
        prompt: str,
        service_account_base64: str,
        model: str,
        location: str,
        aspect_ratio: str,
        resolution: str,
        seed: int = 0,  # cache-buster only; intentionally unused (not sent to Gemini)
        backend: str = "generativelanguage",
        custom_model: str = "",
        image: torch.Tensor = None,
        image_2: torch.Tensor = None,
        image_3: torch.Tensor = None,
        image_4: torch.Tensor = None,
        image_5: torch.Tensor = None,
        image_6: torch.Tensor = None,
        image_7: torch.Tensor = None,
        image_8: torch.Tensor = None,
        image_9: torch.Tensor = None,
        image_10: torch.Tensor = None,
        images=None,
        batch_size: int = 1,
    ):
        # INPUT_IS_LIST: scalars arrive as 1-element lists, image slots as
        # lists of tensors — unwrap the former, flatten the latter
        def _scalar(v, default=None):
            if isinstance(v, list):
                return v[0] if v else default
            return v if v is not None else default

        prompt = _scalar(prompt)
        service_account_base64 = _scalar(service_account_base64, "")
        model = _scalar(model)
        location = _scalar(location)
        aspect_ratio = _scalar(aspect_ratio)
        resolution = _scalar(resolution)
        backend = _scalar(backend, "generativelanguage")
        custom_model = _scalar(custom_model, "")
        batch_size = max(1, int(_scalar(batch_size, 1)))

        if model == "custom" and custom_model:
            model = custom_model

        # key resolution: widget > env var > pack-root .env (gitignored) —
        # leave the widget empty to keep the key out of exported workflows
        from ...common.secrets import get_secret
        service_account_base64 = get_secret("SERVICE_ACCOUNT_BASE64",
                                            service_account_base64)
        if not service_account_base64:
            raise RuntimeError(
                "jz Gemini Generate: no service account — set the "
                "service_account_base64 input, the SERVICE_ACCOUNT_BASE64 env "
                "var, or SERVICE_ACCOUNT_BASE64= in comfyui-jz/.env")

        # Collect provided images. All slots are optional: with zero images this
        # is a pure text-to-image call. A connected-but-empty tensor (0 px, e.g.
        # an unselected LoadImage or an empty crop upstream) would crash the PNG
        # encoder, so skip empties and name the slot in the log.
        def _nonempty(t) -> bool:
            return t is not None and t.ndim == 4 and t.shape[1] > 0 and t.shape[2] > 0

        def _tensors(v):
            if v is None:
                return []
            return [t for t in (v if isinstance(v, list) else [v]) if t is not None]

        collected = []
        for idx, slot in enumerate(
            [image, image_2, image_3, image_4, image_5, image_6, image_7,
             image_8, image_9, image_10, images],
            start=1,
        ):
            for t in _tensors(slot):
                if not _nonempty(t):
                    print(f"[Gemini] skipping empty tensor in slot {idx} (0 px)")
                    continue
                for frame in t:  # every batch frame becomes one image part
                    collected.append(frame.unsqueeze(0))
        images = collected

        # Build parts: text prompt + all images as inline_data
        parts = [{"text": prompt}]
        for img in images:
            img_b64 = self._tensor_to_base64(img)
            parts.append({"inline_data": {"mime_type": "image/png", "data": img_b64}})

        # Authenticate via service account and pick the endpoint. The Vertex AI
        # path (aiplatform) requires the project to have Vertex access to the
        # publisher model; many keys only have AI Studio (generativelanguage)
        # access, so that is the default.
        if backend == "vertex":
            access_token, project_id = _get_access_token(
                service_account_base64, VERTEX_SCOPE
            )
            url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model}:generateContent"
            safety_settings = VERTEX_SAFETY_SETTINGS
        else:
            access_token, _ = _get_access_token(service_account_base64, GLA_SCOPE)
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            safety_settings = GLA_SAFETY_SETTINGS

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        # NOTE: the REST API key is camelCase "safetySettings"; the snake_case
        # form is silently ignored (leaving default safety filters on).
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution,
                },
            },
            "safetySettings": safety_settings,
        }

        from ...common.http import post_with_retries

        def _one_call(i: int) -> tuple:
            # token minted once and shared (read-only headers); each call gets
            # its own transport retries via the pooled session
            resp = post_with_retries(url, headers, payload, timeout=600,
                                     tag=f"jz gemini {i + 1}/{batch_size}")
            if not resp.ok:
                body = re.sub(r'"data"\s*:\s*"[A-Za-z0-9+/=]{100,}"',
                              '"data": "<base64 truncated>"', resp.text)
                raise RuntimeError(f"Gemini API error {resp.status_code}: {body[:500]}")
            data = resp.json()
            try:
                parts_resp = data["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(
                    f"Unexpected response structure: {json.dumps(data)[:500]}")
            usage = data.get("usageMetadata") or {}
            for part in parts_resp:
                if "inlineData" in part:
                    raw = base64.b64decode(part["inlineData"]["data"])
                    return Image.open(BytesIO(raw)).convert("RGB"), usage
            raise RuntimeError(
                f"No image in response parts: {[list(p.keys()) for p in parts_resp]}")

        if batch_size == 1:
            calls = [_one_call(0)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(batch_size, 4)) as pool:
                futs = [pool.submit(_one_call, i) for i in range(batch_size)]
                calls, errors = [], []
                for f in futs:
                    try:
                        calls.append(f.result())
                    except Exception as e:  # noqa: BLE001
                        errors.append(e)
            if not calls:
                raise errors[0]
            if errors:
                print(f"[Gemini] {len(errors)}/{batch_size} batch calls failed: "
                      f"{errors[0]}", flush=True)
        pils = [c[0] for c in calls]
        usages = [c[1] for c in calls]
        total_tokens = sum(u.get("totalTokenCount", 0) for u in usages)
        usage_json = json.dumps({
            "calls": len(usages),
            "prompt_tokens": sum(u.get("promptTokenCount", 0) for u in usages),
            "output_tokens": sum(u.get("candidatesTokenCount", 0) for u in usages),
            "total_tokens": total_tokens,
            "per_call": usages,
        }, ensure_ascii=False)

        # stack into one IMAGE batch; identical aspect/resolution should give
        # identical dims, but resize stragglers to the first frame if not
        ref = pils[0].size
        frames = []
        for p in pils:
            if p.size != ref:
                p = p.resize(ref, Image.LANCZOS)
            arr = np.array(p).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(arr))
        output_tensor = torch.stack(frames, dim=0)

        return (output_tensor, usage_json, total_tokens)


# key frozen: saved workflows reference "GeminiImageGenerate" — never change it
NODE_CLASS_MAPPINGS = {
    "GeminiImageGenerate": jz_GeminiGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageGenerate": "jz Gemini Generate",
}
