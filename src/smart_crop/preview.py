"""Downsized, base64-encoded preview for sending to the vision model.

The full-res original is never sent to the LLM -- only used for the final crop,
via the normalized cx/cy/scale coordinates the model returns.
"""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

PREVIEW_LONG_EDGE = 1280


def preview_data_url(image_path: Path, long_edge: int = PREVIEW_LONG_EDGE) -> str:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        scale = long_edge / max(img.width, img.height)
        if scale < 1.0:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

    return f"data:image/jpeg;base64,{encoded}"
