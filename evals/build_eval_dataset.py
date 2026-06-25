"""
Combine dataset.csv + negative_dataset.csv, resolve chunk_ids to full context text.
Outputs eval_dataset.jsonl — upload this to Colab to run inference manually.

Usage:
    uv run python evals/build_eval_dataset.py
"""
import csv
import json
import random
from pathlib import Path

CHUNKS_PATH = Path(__file__).parent.parent / "retrieval" / "data" / "chunks.json"
POS_CSV = Path(__file__).parent / "dataset.csv"
NEG_CSV = Path(__file__).parent / "negative_dataset.csv"
OUTPUT = Path(__file__).parent / "eval_dataset.jsonl"

rng = random.Random(42)

chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
chunk_by_id = {c["chunk_id"]: c for c in chunks}

by_article: dict = {}
for c in chunks:
    by_article.setdefault(c["article_id"], []).append(c)
article_ids = list(by_article.keys())

rows = []

with open(POS_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        chunk = chunk_by_id.get(row["chunk_id"])
        if not chunk:
            print(f"  MISSING chunk_id: {row['chunk_id']}")
            continue
        context = f"[Article: {chunk['article_title']} | URL: {chunk['article_url']}]\n{chunk['text']}"
        rows.append({
            "question": row["question"],
            "context": context,
            "ground_truth": row["answer"],
            "question_type": "positive",
        })

with open(NEG_CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        source_chunk = chunk_by_id.get(row["chunk_id"])
        source_article_id = source_chunk["article_id"] if source_chunk else None
        other_ids = [aid for aid in article_ids if aid != source_article_id]
        off_topic = rng.choice(by_article[rng.choice(other_ids)])
        context = f"[Article: {off_topic['article_title']} | URL: {off_topic['article_url']}]\n{off_topic['text']}"
        rows.append({
            "question": row["question"],
            "context": context,
            "ground_truth": row["answer"],
            "question_type": "negative",
        })

with open(OUTPUT, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote {len(rows)} rows ({sum(1 for r in rows if r['question_type'] == 'positive')} positive, {sum(1 for r in rows if r['question_type'] == 'negative')} negative)")
print(f"  -> {OUTPUT}")
