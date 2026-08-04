# comfyui-jz

my personal comfyui nodes. everything sits in the `jz/` category so my stuff
never mixes with the installed packs.

the gemini nodes came from ComfyUI-Outpainting-Gemini and kept their class
names, so old workflows load fine. `jz/gemini` has the generate node (vertex
or generativelanguage, works with zero images for text-to-image, takes single
images, batches, or a proper image list), the composite-back node, the pad
calculator and a couple of outpaint helpers. `jz/llm` is one openrouter
vision node that actually raises on errors instead of passing them downstream
as strings, retries on 429/5xx, and downscales images before upload so
openrouter stops 413ing. `jz/util` has a string picker (random or indexed),
a lazy fallback switch that skips the unused branch entirely, and a list
resizer that normalizes mixed-size images to one long edge.

keys never go in workflows. leave the key widgets empty and the nodes look
for them server-side: env vars first (`OPENROUTER_API_KEY`,
`SERVICE_ACCOUNT_BASE64`), then `.env` or `config.ini` at the pack root.
both files are gitignored, `.env.example` shows the format.

adding a node: drop a `.py` under `nodes/` that exports
`NODE_CLASS_MAPPINGS`, the root init finds it. shared retry/session code
lives in `common/http.py`, image helpers in `common/images.py` — use them,
don't copy them.

one rule when touching existing nodes: only append widgets. converting a
widget to a socket or reordering them shifts the saved values in every
workflow using the node (that's how a seed ends up displaying NaN) and the
only cure is deleting and re-adding the node on the canvas.
