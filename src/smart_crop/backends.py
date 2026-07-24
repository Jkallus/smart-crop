"""Named model backends, so different models/servers can be compared on the same images."""

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class Backend:
    name: str
    base_url_env: str
    api_key_env: str
    model: str
    default_base_url: str = "http://127.0.0.1:8000/v1"
    extra_body: dict = None  # extra request params, e.g. disabling Qwen3's default thinking mode

    def __post_init__(self):
        if self.extra_body is None:
            object.__setattr__(self, "extra_body", {})


BACKENDS: dict[str, Backend] = {
    # enable_thinking=False: this task is a structured tool call, not a reasoning problem --
    # Qwen3's default thinking mode was burning ~20k+ tokens of chain-of-thought per image
    # (60-80s/call) for no measurable decision-quality benefit. See agent_spec.md.
    "qwen": Backend(
        "qwen", "OMLX_BASE_URL", "OMLX_API_KEY", "Qwen3.6-35B-A3B-MLX-8bit",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ),
    "gemma": Backend("gemma", "OMLX_BASE_URL", "OMLX_API_KEY", "gemma-4-31B-it-MLX-8bit"),
}


def client_for(backend: Backend) -> OpenAI:
    return OpenAI(
        base_url=os.environ.get(backend.base_url_env, backend.default_base_url),
        api_key=os.environ[backend.api_key_env],
    )
