"""GSR Retrieval Methods: GSRRetrieval + HybridGSR.

Implements the full GSR pipeline (§4 of overall_idea.md):
  Query Q → FAISS candidates → Joint Scoring (text + entity + constraint) → Top-K

And the hybrid variant: GSR + BM25 with Reciprocal Rank Fusion (RRF).

No dependency on g4k — uses gsr_cacl.core.Document directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gsr_cacl.core import Document
from gsr_cacl.kg.builder import build_kg_from_markdown
from gsr_cacl.kg.data_structures import ConstraintKG
from gsr_cacl.encoders.gat_encoder import GATEncoder
from gsr_cacl.scoring.constraint_score import compute_constraint_score
from gsr_cacl.scoring.joint_scorer import JointScorer
from gsr_cacl.datasets.gsr_document import extract_table


class GSRRetrieval:
    """
    Graph-Structured Retrieval (GSR) for financial documents.

    Pipeline per overall_idea.md §4.1:
      1. Build FAISS index over document embeddings  (text signal)
      2. Build Constraint KG for every document       (structure signal)
      3. At query time:
           a. FAISS → candidate set (4×top_k)
           b. Joint Score = α·sim_text + β·sim_entity + γ·CS(G_D)
           c. Return top-K by joint score
    """

    def __init__(
        self,
        corpus: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        candidate_mode: str = "faiss",
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        epsilon: float = 1e-4,
        gat_hidden_dim: int = 256,
        gat_num_heads: int = 4,
        gat_num_layers: int = 2,
        device: str | None = None,
        checkpoint_path: str | None = None,
    ):
        self.corpus = corpus
        self.embedding_function = embedding_function
        self.top_k = top_k
        self.candidate_mode = candidate_mode
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.candidate_mode not in {"faiss", "bruteforce"}:
            raise ValueError(
                f"Unknown candidate_mode: {self.candidate_mode}. "
                "Choose from {'faiss', 'bruteforce'}."
            )

        # Step 1: text candidate store
        if self.candidate_mode == "faiss":
            self._build_faiss_index()
        else:
            self._build_bruteforce_index()

        # Infer embedding dimension from the first document
        self._embed_dim = len(self.doc_text_embeds[0]) if len(self.doc_text_embeds) > 0 else 768

        # Step 2: Constraint KGs for all docs
        self._build_all_kgs()

        # GAT encoder (§4.3)
        self.gat_encoder = GATEncoder(
            embed_dim=self._embed_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=gat_num_heads,
            num_layers=gat_num_layers,
        ).to(self.device)

        # Joint scorer (§4.4)
        self.scorer = JointScorer(
            text_embed_dim=self._embed_dim,
            kg_embed_dim=gat_hidden_dim,
        ).to(self.device)

        # Load pre-trained weights if available
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        self.gat_encoder.eval()
        self.scorer.eval()

        # Pre-encode KGs
        self._encode_all_kgs()

    def _load_checkpoint(self, path: str) -> None:
        """Load pre-trained scorer and GAT encoder weights."""
        import logging
        logger = logging.getLogger(__name__)
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "scorer_state" in ckpt:
            self.scorer.load_state_dict(ckpt["scorer_state"])
            logger.info(f"Loaded scorer weights from {path}")
        if "gat_encoder_state" in ckpt:
            self.gat_encoder.load_state_dict(ckpt["gat_encoder_state"])
            logger.info(f"Loaded GAT encoder weights from {path}")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_faiss_index(self) -> None:
        """Build a direct FAISS index over normalized corpus embeddings."""
        import faiss

        self._id_to_idx: dict[str, int] = {str(doc.id): i for i, doc in enumerate(self.corpus)}
        self._build_text_embeddings()

        # Cosine similarity via normalized vectors + inner product.
        faiss.normalize_L2(self.doc_text_embeds)
        dim = self.doc_text_embeds.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.doc_text_embeds)

    def _build_bruteforce_index(self) -> None:
        """Build the brute-force text store without a FAISS index."""
        self._id_to_idx = {str(doc.id): i for i, doc in enumerate(self.corpus)}
        self._build_text_embeddings()

    def _build_text_embeddings(self) -> None:
        """Encode the whole corpus once so both FAISS and brute force can reuse it."""
        texts = [doc.page_content for doc in self.corpus]
        if hasattr(self.embedding_function, "embed_documents"):
            all_text_embeds = self.embedding_function.embed_documents(texts)
        else:
            all_text_embeds = [self.embedding_function.embed_query(text) for text in texts]

        self.doc_text_embeds = np.asarray(all_text_embeds, dtype="float32")
        if self.doc_text_embeds.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {self.doc_text_embeds.shape}")

    def _build_all_kgs(self) -> None:
        """Build a Constraint KG for every corpus document (§4.2)."""
        self.doc_kgs: list[ConstraintKG] = []
        for doc in tqdm(self.corpus, desc="Building constraint KGs"):
            table_md = self._extract_table(doc.page_content)
            kg = build_kg_from_markdown(table_md)
            self.doc_kgs.append(kg)

    def _encode_all_kgs(self) -> None:
        """Encode all KGs with GAT and cache graph-level embeddings (§4.3)."""
        self.kg_embeds: list[torch.Tensor] = []
        for kg in tqdm(self.doc_kgs, desc="Encoding KGs with GAT"):
            with torch.no_grad():
                doc_embed = self.gat_encoder.encode_graph(kg)
                self.kg_embeds.append(doc_embed)

    # ------------------------------------------------------------------
    # Retrieval (§4.1 + §4.4)
    # ------------------------------------------------------------------

    def _rank_candidates(
        self,
        query: str,
        query_meta: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Return ranked documents with their final retrieval scores."""
        q_vec = np.asarray(self.embedding_function.embed_query(query), dtype="float32").reshape(1, -1)
        import faiss
        faiss.normalize_L2(q_vec)

        k = min(self.top_k * 4, len(self.corpus))
        if self.candidate_mode == "faiss":
            _, indices_np = self.faiss_index.search(q_vec, k)
            candidate_indices = [int(i) for i in indices_np[0] if int(i) >= 0]
        else:
            brute_scores = np.dot(self.doc_text_embeds, q_vec[0])
            candidate_indices = list(np.argsort(brute_scores)[::-1][:k])

        if not candidate_indices:
            return []

        q_text_emb = torch.tensor(q_vec, dtype=torch.float32, device=self.device)

        doc_text_batch = torch.tensor(
            self.doc_text_embeds[candidate_indices],
            dtype=torch.float32,
            device=self.device,
        )
        kg_batch = torch.stack([self.kg_embeds[idx] for idx in candidate_indices])
        constraint_feats = JointScorer.build_constraint_features(
            [compute_constraint_score(self.doc_kgs[idx], epsilon=self.epsilon) for idx in candidate_indices],
            self.device,
        )

        if query_meta is None:
            entity_scores = torch.full((len(candidate_indices),), 0.5, device=self.device)
        else:
            q_vals = {
                key: str(query_meta.get(key, "")).strip().lower()
                for key in ("company_name", "report_year", "company_sector")
            }
            entity_scores = []
            for idx in candidate_indices:
                doc_meta = self.corpus[idx].meta_data
                score = 0.0
                for key in ("company_name", "report_year", "company_sector"):
                    d_val = str(doc_meta.get(key, "")).strip().lower()
                    if q_vals[key] and d_val and q_vals[key] == d_val:
                        score += 1.0 / 3.0
                entity_scores.append(score)
            entity_scores = torch.tensor(entity_scores, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            s_text = self.scorer.forward_text_sim(q_text_emb.repeat(len(candidate_indices), 1), doc_text_batch, kg_batch)
            s_constraint = self.scorer.forward_constraint(constraint_feats)
            final_scores = self.scorer.alpha * s_text + self.scorer.beta * entity_scores + self.scorer.gamma * s_constraint

        scored = list(zip(candidate_indices, final_scores.detach().cpu().tolist()))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)[: self.top_k]
        return [(self.corpus[idx], score) for idx, score in ranked]

    def retrieve(self, query: str, query_meta: dict[str, Any] | None = None) -> list[Document]:
        """Retrieve top-K documents for a query using joint scoring."""
        return [doc for doc, _ in self._rank_candidates(query, query_meta=query_meta)]

    def retrieve_with_scores(
        self,
        query: str,
        query_meta: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve top-K documents and keep their final scores."""
        return self._rank_candidates(query, query_meta=query_meta)

    def retrieve_batch(
        self,
        queries: list[str],
        queries_meta: list[dict[str, Any] | None] | None = None,
        return_scores: bool = False,
    ) -> list[list[Document]] | list[list[tuple[Document, float]]]:
        """Retrieve for a batch of queries."""
        if queries_meta is None:
            queries_meta = [None] * len(queries)
        results = []
        for q, meta in tqdm(
            zip(queries, queries_meta), total=len(queries), desc="GSR Retrieval"
        ):
            if return_scores:
                results.append(self.retrieve_with_scores(q, query_meta=meta))
            else:
                results.append(self.retrieve(q, query_meta=meta))
        return results

    # ------------------------------------------------------------------
    # Entity matching (§4.4)
    # ------------------------------------------------------------------

    def _compute_entity_score(
        self, query_meta: dict[str, Any] | None, doc_idx: int
    ) -> float:
        """
        Entity matching: company + year + sector.
        s_entity = match(company_Q, company_D) + match(year_Q, year_D) + match(sector_Q, sector_D)
        Each match contributes 1/3 when matched.
        """
        if query_meta is None:
            return 0.5  # neutral if no metadata

        doc_meta = self.corpus[doc_idx].meta_data
        score = 0.0
        for key in ("company_name", "report_year", "company_sector"):
            q_val = str(query_meta.get(key, "")).strip().lower()
            d_val = str(doc_meta.get(key, "")).strip().lower()
            if q_val and d_val and q_val == d_val:
                score += 1.0 / 3.0
        return score

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_table(content: str) -> str:
        """Extract the first markdown table from page content."""
        return extract_table(content)


class HybridGSR(GSRRetrieval):
    """
    Hybrid retrieval: GSR + BM25 with Reciprocal Rank Fusion (RRF).

    RRF score = 1/(k + rank_GSR) + 1/(k + rank_BM25)
    """

    def __init__(
        self,
        corpus: list[Document],
        embedding_function: Any,
        top_k: int = 5,
        rrf_k: int = 60,
        **kwargs,
    ):
        super().__init__(
            corpus=corpus,
            embedding_function=embedding_function,
            top_k=top_k,
            **kwargs,
        )
        self.rrf_k = rrf_k
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        self.tokenized_corpus = [
            doc.page_content.split(" ") for doc in self.corpus
        ]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            self._bm25_mode = "bm25"
        except ImportError:
            self.bm25 = None
            self._bm25_mode = "overlap"

    def _rank_candidates(
        self,
        query: str,
        query_meta: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        # GSR ranking
        gsr_scored = super()._rank_candidates(query, query_meta=query_meta)
        gsr_ranking = {d.id: rank for rank, (d, _) in enumerate(gsr_scored)}

        # BM25 ranking or token-overlap fallback
        tokenized_query = query.split(" ")
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(bm25_scores)[::-1][: self.top_k * 4]
        else:
            q_tokens = {t.lower() for t in tokenized_query if t}
            overlap_scores = []
            for idx, doc_tokens in enumerate(self.tokenized_corpus):
                d_tokens = {t.lower() for t in doc_tokens if t}
                overlap = len(q_tokens & d_tokens)
                overlap_scores.append((idx, overlap))
            top_n_indices = [idx for idx, _ in sorted(overlap_scores, key=lambda x: x[1], reverse=True)[: self.top_k * 4]]
        bm25_ranking = {
            self.corpus[i].id: rank for rank, i in enumerate(top_n_indices)
        }

        # Merge candidates
        all_docs: dict[str, Document] = {d.id: d for d, _ in gsr_scored}
        for idx in top_n_indices:
            all_docs[self.corpus[idx].id] = self.corpus[idx]

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        for doc_id in all_docs:
            rrf = 0.0
            if doc_id in gsr_ranking:
                rrf += 1.0 / (self.rrf_k + gsr_ranking[doc_id] + 1)
            if doc_id in bm25_ranking:
                rrf += 1.0 / (self.rrf_k + bm25_ranking[doc_id] + 1)
            rrf_scores[doc_id] = rrf

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            : self.top_k
        ]
        return [(all_docs[doc_id], score) for doc_id, score in ranked]


GSR_REGISTRY: dict[str, type] = {
    "gsr": GSRRetrieval,
    "hybridgsr": HybridGSR,
}
