"""OpenAI client wrapper."""
from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily instantiate the OpenAI SDK client using the env var key."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var missing")
        _client = OpenAI(api_key=api_key)
    return _client


def call_json_model(model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Send a JSON-mode request and parse its structured output."""
    client = get_client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.output[0].content[0].text
    return json.loads(content)
