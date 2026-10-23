"""Style templates for classroom pictures, applied when a job asks for a style.

The same wording the Open WebUI pipe uses (see docs/LEARNINGS.md section 8):
fixed templates keep prompts faithful to the request, which small LLMs
rewriting prompts freely did not.
"""

from __future__ import annotations

import re

STYLES = {
    "icon": "{subject}, flat vector icon, bold clean black outlines, simple shapes, solid bright "
    "colors, centered, plain white background, no text",
    "coloring": "{subject}, coloring book page for kids, black and white line art, thick clean "
    "outlines, no shading, no gray, plain white background, no text",
    "storybook": "children's picture book illustration: {subject}, warm gentle colors{style}, no text",
}
NAMED_STYLE_RE = re.compile(
    r"watercolor|water colou?r|pencil|crayon|pastel|cartoon|painting|painted|digital|3d|pixel|"
    r"photo|realistic|ink|collage|clay|gouache|oil",
    re.I,
)


def build_prompt(style: str | None, subject: str) -> str:
    """The prompt sent to the model; 'none' (or no style) passes the words through."""
    subject = subject.strip()
    template = STYLES.get(style or "none")
    if template is None:
        return subject
    extra = "" if NAMED_STYLE_RE.search(subject) else ", soft watercolor style"
    return template.format(subject=subject.rstrip("."), style=extra)
