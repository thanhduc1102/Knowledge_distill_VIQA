"""DenseExpert — bi-encoder semantic retrieval (e5-large-instruct).

Independent method #2. Single-vector dense retrieval (cosine over 1024-d embeddings).
Weak alone on this benchmark (table not understood, numbers destroy embeddings, same-company
confusion) but contributes *semantic* recall the sparse channel misses; the learned fusion
decides how much to trust it per query. Optional (needs the model + GPU).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gsr_cacl.core import Document
from gsr_cacl.experts.base import Expert

_INSTR = "Given a question about a company, retrieve relevant passages that answer the query."


class DenseExpert(Expert):
    name = "dense"
    is_retriever = True

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct",
                 device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

    def _load(self):
        if getattr(self, "_model", None) is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = SentenceTransformer(self.model_name, device=dev)

    def prepare(self, corpus: Sequence[Document], doc_metas: Sequence[dict]) -> None:
        self._load()
        self._doc_emb = np.asarray(self._model.encode(
            [d.page_content for d in corpus], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, batch_size=self.batch_size), np.float64)

    def set_queries(self, raw_queries: Sequence[str], query_metas: Sequence[dict]) -> None:
        self._q_emb = np.asarray(self._model.encode(
            [f"Instruct: {_INSTR}\nQuery: {q}" for q in raw_queries], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, batch_size=self.batch_size), np.float64)

    def full_scores(self, qi: int) -> np.ndarray:
        return self._doc_emb @ self._q_emb[qi]

    def score_pool(self, qi: int, pool: Sequence[int]) -> np.ndarray:
        return self._doc_emb[np.asarray(pool, dtype=int)] @ self._q_emb[qi]
