"""Secret resolution for nodes: explicit input > process env > pack-root .env.

Keeps keys out of workflow JSON: leave the node's widget empty and the value
is resolved server-side at execution time.
"""
import os
from pathlib import Path

_PACK_ROOT = Path(__file__).resolve().parents[1]


def _read_dotenv() -> dict:
    env_file = _PACK_ROOT / ".env"
    values = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def get_secret(name: str, node_input: str = "") -> str:
    """Resolve a secret: node input > os.environ > .env file. "" if absent."""
    if node_input and node_input.strip():
        return node_input.strip()
    if os.environ.get(name, "").strip():
        return os.environ[name].strip()
    return _read_dotenv().get(name, "")


def openrouter_key(node_input: str = "") -> str:
    """OpenRouter key: node input > env > .env > config.ini [API]. Raises if absent.

    config.ini is checked last and is the historical home for this one key, so
    both OpenRouter nodes resolve it the same way.
    """
    key = get_secret("OPENROUTER_API_KEY", node_input)
    if not key:
        import configparser
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read(str(_PACK_ROOT / "config.ini"), encoding="utf-8")
        key = cfg.get("API", "OPENROUTER_API_KEY", fallback="").strip()
    if not key:
        raise RuntimeError(
            "No OpenRouter key: set the api_key input, the OPENROUTER_API_KEY "
            "env var, or [API] OPENROUTER_API_KEY in comfyui-jz/config.ini "
            "(pack root)")
    return key
