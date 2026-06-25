"""
LLM-as-a-judge eval for the workflow chatbot responder.

For positive examples (dataset.csv):
  - Injects the ground-truth chunk as context
  - Generates a response with RESPONDER_SYSTEM
  - Judges: does the response correctly answer the question?

For negative examples (negative_dataset.csv):
  - Injects a random off-topic chunk as context (simulates bad retrieval)
  - Generates a response with RESPONDER_SYSTEM
  - Judges: does the model correctly refuse instead of hallucinating?

Output: evals/judge_results.csv

Usage:
    uv run python retrieval/evals/judge_eval.py
    uv run python retrieval/evals/judge_eval.py --workers 12
"""
import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
import requests as _requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

CHUNKS_PATH = Path(__file__).parent.parent / "retrieval" / "data" / "chunks.json"
POS_DATASET = Path(__file__).parent / "dataset.csv"
NEG_DATASET = Path(__file__).parent / "negative_dataset.csv"
OUTPUT_PATH = Path(__file__).parent / "judge_results.csv"

RESPONDER_MODEL = "gpt-4.1-nano"
JUDGE_MODEL = "gpt-4o-mini"

RESPONDER_IN  = 0.10 / 1_000_000
RESPONDER_OUT = 0.40 / 1_000_000
JUDGE_IN      = 0.15 / 1_000_000
JUDGE_OUT     = 0.60 / 1_000_000

RESPONDER_SYSTEM = """You are a Vagaro customer support agent. Vagaro is a business management platform for salons, spas, and fitness businesses.

You will be given retrieved help center chunks and a customer question. Answer using only the provided context.

Rules:
- Be concise and actionable
- Always end your answer with "Source: <url>" for every article you draw from
- If the context does not clearly answer the question, say "I don't have enough information to answer that confidently." Do not make things up."""

JUDGE_POSITIVE_SYSTEM = """You are evaluating a customer support AI response for correctness.

You will be given:
- A customer question
- A ground truth answer
- The AI's generated response

Mark the response CORRECT (true) if it covers the central idea and would genuinely help the customer answer their question — even if it uses different wording, includes extra detail, or omits minor specifics.

Mark it INCORRECT (false) ONLY if it:
- Contradicts the ground truth on a key fact
- Completely misses the main point of the answer
- Would mislead the customer or send them in the wrong direction

Minor wording differences, extra helpful detail, or slight omissions of non-critical info are NOT a reason to mark false.

Respond in this exact JSON format:
{"correct": true/false, "justification": "one sentence"}"""

JUDGE_NEGATIVE_SYSTEM = """You are evaluating a customer support AI response.

You will be given:
- A customer question
- The AI's generated response
- The context the AI was given (which is intentionally off-topic / irrelevant)

Judge whether the AI CORRECTLY REFUSED to answer — meaning it said something like "I don't have enough information" rather than hallucinating an answer from the irrelevant context.

Respond in this exact JSON format:
{"correct": true/false, "justification": "one sentence"}"""


def with_backoff(fn, *args, start=3.0, max_retries=6, **kwargs):
    wait = start
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(wait)
            wait *= 2


def generate_response(question: str, context: str, client: OpenAI, responder_url: str | None = None) -> tuple[str, float]:
    if responder_url:
        r = _requests.post(
            f"{responder_url.rstrip('/')}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": RESPONDER_SYSTEM},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
                "max_new_tokens": 512,
            },
            timeout=800,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], 0.0
    resp = with_backoff(
        client.chat.completions.create,
        model=RESPONDER_MODEL,
        messages=[
            {"role": "system", "content": RESPONDER_SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    cost = resp.usage.prompt_tokens * RESPONDER_IN + resp.usage.completion_tokens * RESPONDER_OUT
    return resp.choices[0].message.content, cost


def judge_positive(question: str, ground_truth: str, response: str, client: OpenAI) -> tuple[dict, float]:
    resp = with_backoff(
        client.chat.completions.create,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_POSITIVE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nGround truth: {ground_truth}\n\nGenerated response: {response}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    cost = resp.usage.prompt_tokens * JUDGE_IN + resp.usage.completion_tokens * JUDGE_OUT
    return json.loads(resp.choices[0].message.content), cost


def judge_negative(question: str, context: str, response: str, client: OpenAI) -> tuple[dict, float]:
    resp = with_backoff(
        client.chat.completions.create,
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_NEGATIVE_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nContext given to AI:\n{context}\n\nGenerated response: {response}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    cost = resp.usage.prompt_tokens * JUDGE_IN + resp.usage.completion_tokens * JUDGE_OUT
    return json.loads(resp.choices[0].message.content), cost


def process_positive(row: dict, chunk_by_id: dict, by_article: dict, resp_client: OpenAI, judge_client: OpenAI, responder_url: str | None = None) -> tuple[dict, float]:
    chunk = chunk_by_id.get(row["chunk_id"])
    if not chunk:
        return None, 0.0

    context = f"[Article: {chunk['article_title']} | URL: {chunk['article_url']}]\n{chunk['text']}"
    response, c1 = generate_response(row["question"], context, resp_client, responder_url)
    verdict, c2 = judge_positive(row["question"], row["answer"], response, judge_client)

    return {
        "dataset": "positive",
        "question": row["question"],
        "ground_truth": row["answer"],
        "generated_response": response,
        "correct": verdict["correct"],
        "justification": verdict["justification"],
    }, c1 + c2


def process_negative(row: dict, chunk_by_id: dict, by_article: dict, article_ids: list, rng: random.Random, resp_client: OpenAI, judge_client: OpenAI, responder_url: str | None = None) -> tuple[dict, float]:
    source_chunk = chunk_by_id.get(row["chunk_id"])
    source_article_id = source_chunk["article_id"] if source_chunk else None

    other_ids = [aid for aid in article_ids if aid != source_article_id]
    off_topic = rng.choice(by_article[rng.choice(other_ids)])
    context = f"[Article: {off_topic['article_title']} | URL: {off_topic['article_url']}]\n{off_topic['text']}"

    response, c1 = generate_response(row["question"], context, resp_client, responder_url)
    verdict, c2 = judge_negative(row["question"], context, response, judge_client)

    return {
        "dataset": "negative",
        "question": row["question"],
        "ground_truth": row["answer"],
        "generated_response": response,
        "correct": verdict["correct"],
        "justification": verdict["justification"],
    }, c1 + c2


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers (default: 1 for custom responder, 12 for OpenAI)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pos", type=int, default=None, help="Limit positive examples (default: all)")
    parser.add_argument("--n-neg", type=int, default=None, help="Limit negative examples (default: all)")
    parser.add_argument("--responder-url", type=str, default=None, help="Custom responder base URL (e.g. ngrok). Omit to use OpenAI.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    by_article: dict = {}
    for c in chunks:
        by_article.setdefault(c["article_id"], []).append(c)
    article_ids = list(by_article.keys())
 
    pos_rows = load_csv(POS_DATASET)
    neg_rows = load_csv(NEG_DATASET)
    if args.n_pos:
        pos_rows = rng.sample(pos_rows, min(args.n_pos, len(pos_rows)))
    if args.n_neg:
        neg_rows = rng.sample(neg_rows, min(args.n_neg, len(neg_rows)))
    print(f"Positive examples: {len(pos_rows)}  |  Negative examples: {len(neg_rows)}\n")

    tasks = []
    for row in pos_rows:
        tasks.append(("positive", row))
    for row in neg_rows:
        tasks.append(("negative", row))

    results = []
    done_count = 0
    total_cost = 0.0
    lock = threading.Lock()

    # Responder client: custom URL (Colab/vLLM) or OpenAI
    n_workers = args.workers if args.workers is not None else (1 if args.responder_url else 12)

    if args.responder_url:
        responder_client = None
        print(f"Responder: {args.responder_url}  (judge: {JUDGE_MODEL}, workers: {n_workers})\n")
    else:
        responder_client = None
        print(f"Responder: {RESPONDER_MODEL}  (judge: {JUDGE_MODEL}, workers: {n_workers})\n")

    def run_task(task):
        kind, row = task
        judge_client = OpenAI()
        resp_client = responder_client or OpenAI()
        try:
            if kind == "positive":
                return process_positive(row, chunk_by_id, by_article, resp_client, judge_client, args.responder_url)
            else:
                return process_negative(row, chunk_by_id, by_article, article_ids, random.Random(rng.random()), resp_client, judge_client, args.responder_url)
        except Exception as e:
            return {"dataset": kind, "question": row["question"], "error": str(e)}, 0.0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            result, cost = future.result()
            with lock:
                done_count += 1
                total_cost += cost
                if result and "error" not in result:
                    results.append(result)
                    status = "OK " if result["correct"] else "FAIL"
                    print(f"  [{done_count}/{len(tasks)}] [{result['dataset'].upper()[:3]}] [{status}] {result['question'][:55]}  (${total_cost:.4f})")
                else:
                    err = result.get("error", "unknown") if result else "None returned"
                    print(f"  [{done_count}/{len(tasks)}] SKIP — {err}")

    # Write results
    fieldnames = ["dataset", "question", "ground_truth", "generated_response", "correct", "justification"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(r for r in results if "error" not in r)

    # Summary
    pos_results = [r for r in results if r["dataset"] == "positive"]
    neg_results = [r for r in results if r["dataset"] == "negative"]
    pos_acc = sum(1 for r in pos_results if r["correct"]) / len(pos_results) if pos_results else 0
    neg_acc = sum(1 for r in neg_results if r["correct"]) / len(neg_results) if neg_results else 0
    overall = sum(1 for r in results if r["correct"]) / len(results) if results else 0

    print(f"\n{'='*50}")
    print(f"Positive accuracy  (should answer):  {pos_acc:.1%}  ({len(pos_results)} examples)")
    print(f"Negative accuracy  (should refuse):  {neg_acc:.1%}  ({len(neg_results)} examples)")
    print(f"Overall accuracy:                    {overall:.1%}  ({len(results)} total)")
    print(f"Total cost: ${total_cost:.4f}  ({RESPONDER_MODEL} responses + {JUDGE_MODEL} judge)")
    print(f"Results written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
