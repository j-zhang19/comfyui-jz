"""jz Prompt Builder — fold a VLM scene description into the outpaint prompt,
mirroring the production worker's ``build_prompt`` (fails open to the plain
prompt when the description looks like an upstream error string)."""

PLAIN_PROMPT = (
    "Image 1 has solid colored borders that need to be filled. Image 2 is a mask "
    "where white areas need to be filled and black areas must remain unchanged. "
    "Image 3 is the original image for reference. Seamlessly extend the scene into "
    "the bordered areas.\n\n"
    "Do not alter the composition, subjects, or layout of the original content. "
    "Only generate new content in the solid colored border regions. The generated "
    "content must blend seamlessly at the boundary."
)


_ERROR_MARKERS = ("Error:", "API Error", "Unexpected API response")


class jz_PromptBuilder:
    """Fold a VLM scene description into the outpaint prompt.

    Mirrors ``pipeline.build_prompt``: the description is injected as *context* on
    the same "seamlessly extend" instruction — deliberately NOT as "preserve
    exactly / no visible box" wording, which testing showed makes the model guard
    the original as a rectangle and stamp the very box we want to avoid. When the
    scene context is empty or an error string, it falls back to the plain prompt
    (fail-open), so a describe outage never blocks a job.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scene_context": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = "jz/util"

    def build(self, scene_context: str):
        ctx = (scene_context or "").strip()
        if not ctx or any(ctx.startswith(m) for m in _ERROR_MARKERS):
            return (PLAIN_PROMPT,)

        prompt = (
            "Image 1 has solid colored borders that need to be filled. Image 2 is a mask "
            "where white areas need to be filled and black areas must remain unchanged. "
            "Image 3 is the original image for reference.\n\n"
            f"For context, the scene is: {ctx}\n\n"
            "Seamlessly extend the scene into the bordered areas. Do not alter the "
            "composition, subjects, or layout of the original content. Only generate new "
            "content in the solid colored border regions. The generated content must blend "
            "seamlessly at the boundary."
        )
        return (prompt,)


# key kept as "GeminiPromptBuilder" so saved workflows still resolve
NODE_CLASS_MAPPINGS = {"GeminiPromptBuilder": jz_PromptBuilder}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiPromptBuilder": "jz Prompt Builder (scene-conditioned)"}
