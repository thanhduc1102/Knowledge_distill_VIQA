"""Semantic concept canonicalisation — enrich the Fact Ledger's ontology coverage.

KG construction diagnosis (scripts/research/kg_construction_diag.py): only ~21-25% of facts
receive a canonical concept by exact alias matching, which leaves the accounting-identity
verifier almost dead (identity edges fire on 0-2% of docs). Financial line-items are a long
tail ("provision for credit losses", "gain on extinguishment of debt"), so hand-listing every
alias does not scale.

This module adds an embedding-based fallback: each canonical concept is represented by the
mean embedding of its aliases; an un-canonicalised row label is assigned to its nearest
canonical concept when cosine ≥ ``threshold``. A deliberately HIGH threshold avoids the
finance trap of mapping opposite items that are lexically close (e.g. "interest income" vs
"interest expense"). Used to enrich the typed fact graph (more identity edges → more
verification + provenance), not as a retrieval ranking signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from gsr_cacl.ledger.fact import FactLedger
from gsr_cacl.ontology.concepts import CONCEPT_ALIASES, canonical_concept


class SemanticCanonicalizer:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5",
                 device: Optional[str] = None, threshold: float = 0.72):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self._model = None
        self._concepts: list[str] = []
        self._anchors: Optional[np.ndarray] = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(self.model_name, device=dev)
        # one anchor vector per canonical concept = mean of its alias embeddings
        anchors = []
        for concept, aliases in CONCEPT_ALIASES.items():
            embs = self._model.encode(list(aliases) or [concept], normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
            v = embs.mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-9)
            self._concepts.append(concept)
            anchors.append(v)
        self._anchors = np.asarray(anchors, dtype=np.float32)

    def nearest(self, label: str) -> Optional[str]:
        self._load()
        v = self._model.encode([label], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)[0]
        sims = self._anchors @ v
        j = int(np.argmax(sims))
        return self._concepts[j] if sims[j] >= self.threshold else None

    def enrich(self, ledger: FactLedger) -> FactLedger:
        """Fill missing ``concept_canonical`` on the ledger's facts in place."""
        missing = [f for f in ledger.facts if not f.concept_canonical and (f.concept or "").strip()]
        if not missing:
            return ledger
        self._load()
        labels = [f.concept for f in missing]
        embs = self._model.encode(labels, normalize_embeddings=True, convert_to_numpy=True,
                                  batch_size=256, show_progress_bar=False)
        sims = embs @ self._anchors.T                      # [n_missing, n_concepts]
        for f, row in zip(missing, sims):
            # exact alias first (cheap, precise), then semantic fallback
            exact = canonical_concept(f.concept)
            if exact:
                f.concept_canonical = exact
                continue
            j = int(np.argmax(row))
            if row[j] >= self.threshold:
                f.concept_canonical = self._concepts[j]
        return ledger
