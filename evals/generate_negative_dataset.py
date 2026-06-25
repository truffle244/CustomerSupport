"""
Generate negative eval pairs: questions where the retrieved context will be irrelevant,
so the model should respond "I don't have enough information to answer that confidently."

Each row has a real question (generated from chunk A) but the expected answer is the
refusal string — used to eval whether the model correctly refuses when context is off-topic.

Usage:
    uv run python retrieval/evals/generate_negative_dataset.py
    uv run python retrieval/evals/generate_negative_dataset.py --n 50 --seed 42
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
sys.path.insert(0, str(Path(__file__).parent))

from generate_dataset import generate_qa, PARAPHRASE_RATE  # noqa: E402

CHUNKS_PATH = Path(__file__).parent.parent / "retrieval" / "data" / "chunks.json"
OUTPUT_PATH = Path(__file__).parent / "negative_dataset.csv"

MODEL = "gpt-4.1-nano"
INPUT_COST_PER_1M = 0.10
OUTPUT_COST_PER_1M = 0.40

REFUSAL = "I don't have enough information to answer that confidently."


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


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["chunk_id"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = [c for c in chunks if c["text"].strip()]

    existing_ids = load_existing_ids(OUTPUT_PATH)
    candidates = [c for c in chunks if c["chunk_id"] not in existing_ids]
    if not candidates:
        print("No new chunks to generate from.")
        sys.exit(0)

    selected = random.sample(candidates, min(args.n, len(candidates)))
    client = OpenAI()
    total_input = total_output = 0
    rows = []

    print(f"Generating {len(selected)} negative eval pairs with {MODEL}...\n")

    for i, chunk in enumerate(selected, 1):
        try:
            paraphrase = random.random() < PARAPHRASE_RATE
            qa, inp, out = with_backoff(generate_qa, chunk, client, paraphrase=paraphrase)
            total_input += inp
            total_output += out
        except Exception as e:
            print(f"  [{i}/{len(selected)}] SKIP — {e}")
            continue

        cost = (total_input / 1e6 * INPUT_COST_PER_1M) + (total_output / 1e6 * OUTPUT_COST_PER_1M)
        print(f"  [{i}/{len(selected)}] {chunk['article_title'][:55]}  (${cost:.5f})")

        rows.append({
            "question": qa["question"],
            "chunk_id": chunk["chunk_id"],
            "answer": REFUSAL,
            "question_type": "negative",
        })

    write_header = not OUTPUT_PATH.exists()
    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "chunk_id", "answer", "question_type"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    total_cost = (total_input / 1e6 * INPUT_COST_PER_1M) + (total_output / 1e6 * OUTPUT_COST_PER_1M)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_PATH}")
    print(f"Cost: ${total_cost:.5f}  ({total_input:,} input + {total_output:,} output tokens)")


if __name__ == "__main__":
    main()
