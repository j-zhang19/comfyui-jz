"""Secret resolution for nodes: explicit input > process env > pack-root .env.

Keeps keys out of workflow JSON: leave the node's widget empty and the value
is resolved server-side at execution time.
"""
import configparser
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


def resolve_key(env_names: list, ini: tuple, node_input: str = "",
                hint: str = "") -> str:
    """First hit wins: node input > env vars (process env, then .env) >
    config.ini. Raises `hint` when nothing is set.

    Every provider key in this pack resolves the same way; only the names and
    the config.ini section differ.
    """
    if node_input and node_input.strip():
        return node_input.strip()
    for name in env_names:
        found = get_secret(name)
        if found:
            return found
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(str(_PACK_ROOT / "config.ini"), encoding="utf-8")
    found = cfg.get(ini[0], ini[1], fallback="").strip()
    if found:
        return found
    raise RuntimeError(hint)


def openrouter_key(node_input: str = "") -> str:
    return resolve_key(
        ["OPENROUTER_API_KEY"], ("API", "OPENROUTER_API_KEY"), node_input,
        "No OpenRouter key: set the api_key input, the OPENROUTER_API_KEY env "
        "var, or [API] OPENROUTER_API_KEY in comfyui-jz/config.ini (pack root)")


def openai_key(node_input: str = "") -> str:
    return resolve_key(
        ["OPENAI_API_KEY"], ("OPENAI", "OPENAI_API_KEY"), node_input,
        "No OpenAI key: set the api_key input, OPENAI_API_KEY in "
        "comfyui-jz/.env, or [OPENAI] OPENAI_API_KEY in config.ini")


def byteplus_key(node_input: str = "") -> str:
    return resolve_key(
        ["BYTEPLUS_API_KEY", "ARK_API_KEY"], ("BYTEDANCE", "ARK_API_KEY"),
        node_input,
        "No BytePlus key: set the api_key input, BYTEPLUS_API_KEY in "
        "comfyui-jz/.env, or [BYTEDANCE] ARK_API_KEY in config.ini")
