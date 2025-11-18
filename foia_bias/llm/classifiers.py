"""LLM-powered classifiers."""
from __future__ import annotations

import json
from typing import Any, Dict

from foia_bias.llm.client import call_json_model
from foia_bias.llm.prompts import BASE_SYSTEM_PROMPT, CLASSIFICATION_TEMPLATE


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def classify_document(text: str, doc_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    llm_config = config.get("llm", {})
    model = llm_config.get("classifier_model", "gpt-5.1-thinking")
    max_chars = llm_config.get("max_chars_per_doc", 20000)
    prompt = CLASSIFICATION_TEMPLATE.format(doc_id=doc_id, doc_text=truncate_text(text, max_chars))
    return call_json_model(model, BASE_SYSTEM_PROMPT, prompt)
