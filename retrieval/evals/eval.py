"""
Retrieval eval: Hit Rate@k and MRR@k, dense vs hybrid.
Threaded — all queries run in parallel.

Usage:
    uv run python retrieval/evals/eval.py
    uv run python retrieval/evals/eval.py --mode dense
    uv run python retrieval/evals/eval.py --mode hybrid
    uv run python retrieval/evals/eval.py --ks 1 3 5 10
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from retrieve import get_collection  # noqa: E402

DATASET_PATH = Path(__file__).parent / "dataset.csv"
CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
CHUNKS_PATH = str(Path(__file__).parent.parent / "data" / "chunks.json")
EMBED_MODEL = "text-embedding-3-small"
COST_PER_1M = 0.02


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, newline="", encoding="utf-8") as f:
        return [{"question": r["question"], "chunk_id": r["chunk_id"].strip()} for r in csv.DictReader(f)]


def metrics(results: list[dict], ks: list[int]) -> list[dict]:
    rows = []
    for k in ks:
        hit_rate = sum(1 for r in results if r["rank"] and r["rank"] <= k) / len(results)
        mrr = sum(r["rr"] for r in results if r["rank"] and r["rank"] <= k) / len(results)
        rows.append({"k": k, "Hit Rate": round(hit_rate, 3), "MRR": round(mrr, 3)})
    return rows


def run_dense(dataset: list[dict], ks: list[int], client: OpenAI):
    collection = get_collection(CHROMA_PATH)
    max_k = max(ks)

    # Batched embed calls (max 2048 per request)
    BATCH_SIZE = 2048
    t0 = time.time()
    questions = [r["question"] for r in dataset]
    embeddings, tokens = [], 0
    for i in range(0, len(questions), BATCH_SIZE):
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=questions[i : i + BATCH_SIZE],
            encoding_format="float",
        )
        embeddings += [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        tokens += resp.usage.total_tokens
    print(f"  Embedded {len(dataset)} queries in {time.time()-t0:.1f}s  (${tokens/1_000_000*COST_PER_1M:.6f})")

    # Parallel Chroma queries
    def query(args):
        idx, row, vec = args
        hits = collection.query(query_embeddings=[vec], n_results=max_k, include=[])
        ranked_ids = hits["ids"][0]
        rank = next((j + 1 for j, cid in enumerate(ranked_ids) if cid == row["chunk_id"]), None)
        return idx, {"rank": rank, "rr": 1 / rank if rank else 0.0}

    t0 = time.time()
    results = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(query, (i, row, vec)) for i, (row, vec) in enumerate(zip(dataset, embeddings))]
        for done in as_completed(futures):
            idx, result = done.result()
            results[idx] = result
    print(f"  Chroma queries done in {time.time()-t0:.1f}s")

    return metrics(results, ks)


def run_hybrid(dataset: list[dict], ks: list[int], client: OpenAI):
    from hybrid_retrieve import HybridRetriever
    retriever = HybridRetriever(CHUNKS_PATH, CHROMA_PATH, openai_client=client)
    max_k = max(ks)

    t0 = time.time()
    queries = [r["question"] for r in dataset]
    all_ranked = retriever.batch_retrieve(queries, n_results=max_k, k_candidates=10)
    print(f"  Done in {time.time()-t0:.1f}s")

    results = []
    for row, ranked in zip(dataset, all_ranked):
        ranked_ids = [r["chunk_id"] for r in ranked]
        rank = next((j + 1 for j, cid in enumerate(ranked_ids) if cid == row["chunk_id"]), None)
        results.append({"rank": rank, "rr": 1 / rank if rank else 0.0})

    return metrics(results, ks)


def print_table(rows: list[dict], label: str):
    print(f"\n{label}")
    print(f"  {'k':<6} {'Hit Rate':>10} {'MRR':>10}")
    print(f"  {'-'*28}")
    for r in rows:
        print(f"  {r['k']:<6} {r['Hit Rate']:>10.3f} {r['MRR']:>10.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid", "both"], default="both")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} eval pairs\n")
    client = OpenAI()

    if args.mode in ("dense", "both"):
        print("--- Dense ---")
        dense_rows = run_dense(dataset, args.ks, client)
        print_table(dense_rows, "Dense results")

    if args.mode in ("hybrid", "both"):
        print("\n--- Hybrid ---")
        hybrid_rows = run_hybrid(dataset, args.ks, client)
        print_table(hybrid_rows, "Hybrid results")
