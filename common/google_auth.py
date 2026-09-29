"""Google service-account -> OAuth2 access token.

Signs a JWT with openssl (no extra crypto dependency) and exchanges it. Used by
jz Gemini Generate for both backends, which differ only in the scope they ask
for.
"""
import base64
import json
import os
import subprocess
import tempfile
import time

import requests

GLA_SCOPE = "https://www.googleapis.com/auth/generative-language"
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def access_token(
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
