"""Embedding-based political relevance filter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

from foia_bias.llm.client import get_client


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-large"
    max_chars: int = 2000


class PoliticalRelevanceClassifier:
    """Trainable logistic regression on embeddings for cheap filtering."""
    def __init__(self, emb_config: EmbeddingConfig | None = None):
        self.emb_config = emb_config or EmbeddingConfig()
        self.model = LogisticRegression(max_iter=1000)
        self._is_fit = False

    def _embed(self, text: str) -> List[float]:
        """Generate a single embedding vector using the configured model."""
        client = get_client()
        resp = client.embeddings.create(
            model=self.emb_config.model,
            input=text[: self.emb_config.max_chars],
        )
        return resp.data[0].embedding

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> None:
        """Fit the logistic regression classifier on embeddings."""
        vectors = np.vstack([self._embed(t) for t in texts])
        y = np.array(list(labels))
        self.model.fit(vectors, y)
        self._is_fit = True

    def predict_proba(self, text: str) -> float:
        """Return the probability a text is political, raising if unfitted."""
        if not self._is_fit:
            raise RuntimeError("PoliticalRelevanceClassifier not fit")
        vector = np.array(self._embed(text)).reshape(1, -1)
        return float(self.model.predict_proba(vector)[0, 1])
