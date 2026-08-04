# comfyui-jz

Personal node pack. Everything lives under the `jz/` category.

```
nodes/
  gemini/   outpaint (Vertex/GLA generate + composite), pad calculator, seam repair, prompt builder
  llm/      OpenRouter VLM (vision -> text, retries, cost output)
common/
  http.py   shared transport: pooled session, 408/429/5xx retries, Retry-After
  images.py tensor/PIL/base64 helpers
```

- Adding a node: drop a `.py` exporting `NODE_CLASS_MAPPINGS` anywhere under
  `nodes/` — the root `__init__.py` auto-discovers it.
- Legacy Gemini node class keys are preserved from ComfyUI-Outpainting-Gemini,
  so existing workflows keep working.
- API keys: `config.ini` at pack root (see `config.ini.example`, gitignored)
  or `OPENROUTER_API_KEY` env, or node input.
