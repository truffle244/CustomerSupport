"""
Hybrid retrieval: BM25 + dense vector search fused with RRF, then cross-encoder reranking.

Usage:
    uv run python retrieval/hybrid_retrieve.py "how do I process a refund"

Compared to retrieve.py (dense-only), this adds:
  1. BM25 sparse search over chunks.json
  2. Reciprocal Rank Fusion (RRF) to merge dense + sparse ranked lists
  3. Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2, runs locally)
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from retrieve import get_collection

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF constant — higher = smoother fusion


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class HybridRetriever:
    def __init__(self, chunks_path: str, chroma_path: str, openai_client: OpenAI = None):
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_meta = {c["chunk_id"]: c for c in chunks}

        print("Building BM25 index...")
        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in chunks])

        print("Loading cross-encoder (downloads on first run ~90MB)...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        self.collection = get_collection(chroma_path)
        self.openai_client = openai_client or OpenAI()

    def _dense_search(self, query: str, n: int) -> list[tuple[str, int]]:
        """Returns [(chunk_id, rank), ...] from vector search."""
        resp = self.openai_client.embeddings.create(
            model=EMBED_MODEL, input=[query], encoding_format="float"
        )
        results = self.collection.query(
            query_embeddings=[resp.data[0].embedding],
            n_results=n,
            include=[],
        )
        return [(cid, rank + 1) for rank, cid in enumerate(results["ids"][0])]

    def _bm25_search(self, query: str, n: int) -> list[tuple[str, int]]:
        """Returns [(chunk_id, rank), ...] from BM25 search."""
        scores = self.bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [(self.chunk_ids[i], rank + 1) for rank, i in enumerate(top_indices)]

    def _rrf_fusion(self, ranked_lists: list[list[tuple[str, int]]]) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion across multiple ranked lists."""
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for chunk_id, rank in ranked:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (RRF_K + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _get_candidates(self, query: str, k: int) -> tuple[str, list[dict]]:
        """Embed + BM25 + RRF for one query. Returns (query, candidates) without reranking."""
        dense = self._dense_search(query, k)
        sparse = self._bm25_search(query, k)
        fused = self._rrf_fusion([dense, sparse])
        candidate_ids = [cid for cid, _ in fused[:k]]
        candidates = [self.chunk_meta[cid] for cid in candidate_ids if cid in self.chunk_meta]
        return query, candidates

    def batch_retrieve(
        self,
        queries: list[str],
        n_results: int = 5,
        k_candidates: int = 10,
        embed_workers: int = 8,
    ) -> list[list[dict]]:
        """
        Retrieve + rerank for multiple queries efficiently.
        Parallelizes embed+BM25+RRF, then runs one batched cross-encoder call.
        """
        # Stage 1: parallel candidate fetch (I/O-bound embed + CPU BM25)
        all_candidates: list[list[dict]] = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=embed_workers) as pool:
            futures = {pool.submit(self._get_candidates, q, k_candidates): i for i, q in enumerate(queries)}
            for done in futures:
                i = futures[done]
                _, candidates = done.result()
                all_candidates[i] = candidates

        # Stage 2: one big batched cross-encoder call across all queries
        all_pairs = [(queries[i], c["text"]) for i, cands in enumerate(all_candidates) for c in cands]
        all_scores = self.reranker.predict(all_pairs, batch_size=64) if all_pairs else []

        # Split scores back per query and sort
        results = []
        offset = 0
        for i, candidates in enumerate(all_candidates):
            n = len(candidates)
            scores = all_scores[offset : offset + n]
            offset += n
            reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            results.append([
                {
                    "chunk": c["text"],
                    "source_title": c["article_title"],
                    "source_url": c["article_url"],
                    "article_id": c["article_id"],
                    "chunk_id": c["chunk_id"],
                    "score": float(s),
                }
                for c, s in reranked[:n_results]
            ])
        return results

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        k_candidates: int = 10,
    ) -> list[dict]:
        """
        Hybrid retrieve + rerank.

        k_candidates: how many candidates to pull from each search before fusion/reranking.
        n_results: final number of results to return after reranking.
        """
        dense = self._dense_search(query, k_candidates)
        sparse = self._bm25_search(query, k_candidates)
        fused = self._rrf_fusion([dense, sparse])

        # Take top candidates for reranking
        candidate_ids = [cid for cid, _ in fused[:k_candidates]]
        candidates = [self.chunk_meta[cid] for cid in candidate_ids if cid in self.chunk_meta]

        # Cross-encoder reranking
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = self.reranker.predict(pairs)

        reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)

        return [
            {
                "chunk": c["text"],
                "source_title": c["article_title"],
                "source_url": c["article_url"],
                "article_id": c["article_id"],
                "chunk_id": c["chunk_id"],
                "score": float(score),
            }
            for c, score in reranked[:n_results]
        ]


if __name__ == "__main__":
    import sys

    def safe_print(text: str):
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

    query = " ".join(sys.argv[1:]) or "how do I process a refund"

    retriever = HybridRetriever(
        chunks_path="retrieval/data/chunks.json",
        chroma_path="retrieval/chroma_db",
    )

    results = retriever.retrieve(query, n_results=5)
    safe_print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        safe_print(f"--- Result {i} (rerank score: {r['score']:.4f}) ---")
        safe_print(f"Title: {r['source_title']}")
        safe_print(f"URL:   {r['source_url']}")
        safe_print(f"Chunk: {r['chunk'][:300]}...")
        safe_print("")
