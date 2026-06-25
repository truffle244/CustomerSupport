"""
Fine-tuning dataset generator for the responder stage.

Produces OpenAI fine-tuning JSONL where each example is:
  system: RESPONDER_SYSTEM
  user:   context (source chunk for positives, random off-topic chunk for negatives) + question
  assistant: styled answer with source URL (or refusal for negatives)

Both question gen and response gen use gpt-4.1-nano.
Examples are generated in parallel via ThreadPoolExecutor (LLM calls are I/O bound).

Usage:
    uv run python finetuning/generate_dataset.py
    uv run python finetuning/generate_dataset.py --n 2000 --workers 20
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RETRIEVAL_DIR = Path(__file__).parent.parent / "retrieval"
EVALS_DIR = Path(__file__).parent.parent / "evals"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(EVALS_DIR))

from generate_dataset import generate_qa, PARAPHRASE_RATE  # noqa: E402

CHUNKS_PATH = RETRIEVAL_DIR / "data" / "chunks.json"
OUTPUT_PATH = Path(__file__).parent / "dataset.jsonl"

MODEL = "gpt-4.1-nano"
INPUT_COST_PER_1M = 0.10
OUTPUT_COST_PER_1M = 0.40

REFUSAL = "I don't have enough information to answer that confidently."

RESPONDER_SYSTEM = """You are a Vagaro customer support agent. Vagaro is a business management platform for salons, spas, and fitness businesses.

You will be given retrieved help center chunks and a customer question. Answer using only the provided context.

Rules:
- Be concise and actionable
- Always end your answer with "Source: <url>" for every article you draw from
- If the context does not clearly answer the question, say "I don't have enough information to answer that confidently." Do not make things up."""


def with_backoff(fn, *args, start=3.0, max_retries=6, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on rate limit errors."""
    wait = start
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)
            wait *= 2


def format_context(chunks: list[dict]) -> str:
    parts = [f"[Article: {c['article_title']} | URL: {c['article_url']}]\n{c['text']}" for c in chunks]
    return "\n\n---\n\n".join(parts)


def generate_response(question: str, context: str, client: OpenAI) -> tuple[str, int, int]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RESPONDER_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
    )
    msg = resp.choices[0].message.content
    return msg, resp.usage.prompt_tokens, resp.usage.completion_tokens


def make_example(question: str, context: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": RESPONDER_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            {"role": "assistant", "content": answer},
        ]
    }


def process_one(
    idx: int,
    chunk: dict,
    is_neg: bool,
    by_article: dict,
    article_ids: list,
    client: OpenAI,
    rng: random.Random,
) -> tuple[int, dict | None, int, int, str]:
    """Generate one example. Returns (idx, example_or_None, input_tokens, output_tokens, status)."""
    total_inp = total_out = 0
    tag = "[NEG]" if is_neg else "[POS]"

    try:
        paraphrase = rng.random() < PARAPHRASE_RATE
        qa, inp, out = with_backoff(generate_qa, chunk, client, paraphrase=paraphrase)
        total_inp += inp
        total_out += out
        question = qa["question"]
    except Exception as e:
        return idx, None, 0, 0, f"SKIP — question gen failed: {e}"

    if is_neg:
        other_ids = [aid for aid in article_ids if aid != chunk["article_id"]]
        other_id = rng.choice(other_ids)
        neg_chunk = rng.choice(by_article[other_id])
        context = format_context([neg_chunk])
        answer = REFUSAL
    else:
        context = format_context([chunk])
        try:
            answer, inp, out = with_backoff(generate_response, question, context, client)
            total_inp += inp
            total_out += out
        except Exception as e:
            return idx, None, total_inp, total_out, f"SKIP — response gen failed: {e}"

    status = f"{tag} {chunk['article_title'][:50]}"
    return idx, make_example(question, context, answer), total_inp, total_out, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--neg-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = [c for c in chunks if c["text"].strip()]

    by_article: dict[int, list[dict]] = {}
    for c in chunks:
        by_article.setdefault(c["article_id"], []).append(c)
    article_ids = list(by_article.keys())

    if args.n <= len(chunks):
        selected = rng.sample(chunks, args.n)
    else:
        weights = [c["token_count"] for c in chunks]
        selected = rng.choices(chunks, weights=weights, k=args.n)

    n_neg = max(1, round(len(selected) * args.neg_rate))
    neg_indices = set(rng.sample(range(len(selected)), min(n_neg, len(selected))))

    # Each thread gets its own OpenAI client and RNG (both are not thread-safe)
    def make_worker_args(i, chunk):
        return (
            i, chunk, i in neg_indices, by_article, article_ids,
            OpenAI(),
            random.Random(args.seed + i),
        )

    total_input = total_output = 0
    seen_questions: set[str] = set()
    seen_lock = threading.Lock()
    counter_lock = threading.Lock()
    results: dict[int, dict] = {}
    done = 0

    print(f"Generating {len(selected)} examples ({n_neg} negatives) with {MODEL} — {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, *make_worker_args(i, c)): i for i, c in enumerate(selected)}
        for future in as_completed(futures):
            idx, example, inp, out, status = future.result()
            done += 1

            with counter_lock:
                total_input += inp
                total_output += out
                cost = (total_input / 1e6 * INPUT_COST_PER_1M) + (total_output / 1e6 * OUTPUT_COST_PER_1M)

            if example is None:
                print(f"  [{done}/{len(selected)}] {status}")
                continue

            q_norm = example["messages"][1]["content"].split("Question:")[-1].strip().lower()
            with seen_lock:
                if q_norm in seen_questions:
                    print(f"  [{done}/{len(selected)}] SKIP — duplicate question")
                    continue
                seen_questions.add(q_norm)

            results[idx] = example
            print(f"  [{done}/{len(selected)}] {status}  (${cost:.4f} so far)")

    # Write in original order
    examples = [results[i] for i in sorted(results)]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total_cost = (total_input / 1e6 * INPUT_COST_PER_1M) + (total_output / 1e6 * OUTPUT_COST_PER_1M)
    print(f"\nWrote {len(examples)} examples to {OUTPUT_PATH}")
    print(f"  Model: {MODEL}  |  {total_input:,} input + {total_output:,} output tokens")
    print(f"  Total cost: ${total_cost:.4f}  (~${total_cost/max(len(examples),1)*2000:.2f} projected for 2k)")


if __name__ == "__main__":
    main()
