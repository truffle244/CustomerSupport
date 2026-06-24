"""
Generate question/answer eval pairs from random chunks using gpt-4o-mini.

Usage:
    uv run python retrieval/evals/generate_dataset.py
    uv run python retrieval/evals/generate_dataset.py --n 20 --seed 42
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DATASET_PATH = Path(__file__).parent / "dataset.csv"
CHUNKS_PATH = Path(__file__).parent.parent / "data" / "chunks.json"

MODEL = "gpt-4.1-nano"
# gpt-4o-mini pricing (as of 2025)
INPUT_COST_PER_1M = 0.10
OUTPUT_COST_PER_1M = 0.40

PARAPHRASE_RATE = 0.2  # fraction of questions generated with harder paraphrased prompt

SYSTEM_PROMPT = """You are building a retrieval evaluation dataset for a customer support RAG system.

Given a chunk of text from a Vagaro help center article, generate ONE question that:
- Can be clearly and completely answered using only the given chunk
- A real customer or support agent would plausibly ask
- Is specific enough that only this chunk (not generic knowledge) answers it

Then provide a concise, accurate answer based solely on the chunk.

Respond in this exact JSON format:
{
  "question": "...",
  "answer": "..."
}"""

PARAPHRASE_SYSTEM_PROMPT = """You are building a HARD retrieval evaluation dataset for a customer support RAG system.

Given a chunk of text from a Vagaro help center article, generate ONE question that:
- Can be answered using the given chunk, but uses DIFFERENT vocabulary — no shared keywords with the chunk
- Sounds like something a confused or non-technical customer would say out loud
- Is conversational and indirect, not a keyword search
- Still has a clear, correct answer in the chunk

The goal is to stress-test semantic search — the question should not "word-match" the chunk at all.

Then provide a concise, accurate answer based solely on the chunk.

Respond in this exact JSON format:
{
  "question": "...",
  "answer": "..."
}"""

USER_PROMPT = """Article: {title}
Section: {heading}

Chunk:
{text}

Generate a question and answer pair."""


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["chunk_id"] for row in csv.DictReader(f)}


def generate_qa(chunk: dict, client: OpenAI, paraphrase: bool = False) -> tuple[dict, int, int]:
    system = PARAPHRASE_SYSTEM_PROMPT if paraphrase else SYSTEM_PROMPT
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    title=chunk["article_title"],
                    heading=chunk["heading"] or "Introduction",
                    text=chunk["text"],
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.7 if paraphrase else 0.3,
    )
    usage = response.usage
    result = json.loads(response.choices[0].message.content)
    return result, usage.prompt_tokens, usage.completion_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of new pairs to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    existing_ids = load_existing_ids(DATASET_PATH)

    candidates = [c for c in chunks if c["chunk_id"] not in existing_ids and c["text"].strip()]
    if not candidates:
        print("No new chunks to generate from.")
        sys.exit(0)

    random.seed(args.seed)
    selected = random.sample(candidates, min(args.n, len(candidates)))

    # Cost estimate upfront
    avg_chunk_tokens = sum(c["token_count"] for c in selected) // len(selected)
    est_input = len(selected) * (avg_chunk_tokens + 200)  # +200 for prompts
    est_output = len(selected) * 80
    est_cost = (est_input / 1_000_000 * INPUT_COST_PER_1M) + (est_output / 1_000_000 * OUTPUT_COST_PER_1M)
    print(f"Generating {len(selected)} QA pairs with {MODEL}")
    print(f"Est. cost: ${est_cost:.4f} ({est_input:,} input + {est_output:,} output tokens)\n")

    client = OpenAI()
    rows = []
    total_input = total_output = 0

    for i, chunk in enumerate(selected, 1):
        try:
            paraphrase = random.random() < PARAPHRASE_RATE
            qa, inp, out = generate_qa(chunk, client, paraphrase=paraphrase)
            total_input += inp
            total_output += out
            cost_so_far = (total_input / 1_000_000 * INPUT_COST_PER_1M) + (total_output / 1_000_000 * OUTPUT_COST_PER_1M)
            tag = " [paraphrase]" if paraphrase else ""
            print(f"  [{i}/{len(selected)}] {chunk['article_title'][:50]}{tag} — ${cost_so_far:.5f} so far")
            rows.append({
                "question": qa["question"],
                "chunk_id": chunk["chunk_id"],
                "answer": qa["answer"],
                "question_type": "paraphrase" if paraphrase else "direct",
            })
        except Exception as e:
            print(f"  [{i}/{len(selected)}] FAILED ({chunk['chunk_id']}): {e}")

    # Append to CSV (write header only if file doesn't exist)
    write_header = not DATASET_PATH.exists()
    with open(DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "chunk_id", "answer", "question_type"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    total_cost = (total_input / 1_000_000 * INPUT_COST_PER_1M) + (total_output / 1_000_000 * OUTPUT_COST_PER_1M)
    print(f"\nWrote {len(rows)} rows to {DATASET_PATH}")
    print(f"Actual cost: ${total_cost:.5f} ({total_input:,} input + {total_output:,} output tokens)")


if __name__ == "__main__":
    main()
