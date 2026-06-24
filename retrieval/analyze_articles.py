"""
Fetch all Vagaro articles and report token-length distribution.
Helps decide on chunking thresholds.

Usage:
    uv run python analyze_articles.py
    uv run python analyze_articles.py --use-cache   # read from data/articles.json
"""
import argparse
import json
import time
from pathlib import Path

import requests
import tiktoken

BASE_URL = "https://support.vagaro.com/api/v2/help_center/en-us/articles.json"
ENCODING = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 512  # must match chunk.py


def fetch_all() -> list[dict]:
    articles = []
    url = BASE_URL + "?per_page=30"
    while url:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for a in data["articles"]:
            articles.append({"id": a["id"], "title": a["title"], "body": a["body"]})
        print(f"  Fetched {len(articles)} articles...", end="\r")
        url = data["next_page"]
        time.sleep(0.3)
    print()
    return articles


def percentile(sorted_vals: list[int], p: float) -> int:
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def bucket(token_count: int) -> str:
    if token_count <= MAX_TOKENS:
        return f"<= {MAX_TOKENS} (single chunk)"
    elif token_count <= 2_000:
        return f"{MAX_TOKENS+1}-2000 (section split)"
    elif token_count <= 10_000:
        return "2001-10000 (section split, some windows)"
    else:
        return "> 10000 (heavy window splitting)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    if args.use_cache and Path("data/articles.json").exists():
        articles = json.loads(Path("data/articles.json").read_text())
        print(f"Loaded {len(articles)} articles from cache")
    else:
        print("Fetching all articles from API...")
        articles = fetch_all()

    counts = sorted(len(ENCODING.encode(a["body"])) for a in articles)
    n = len(counts)

    print(f"\n--- Token length distribution ({n} articles) ---")
    print(f"  Min:    {counts[0]}")
    print(f"  p25:    {percentile(counts, 25)}")
    print(f"  Median: {percentile(counts, 50)}")
    print(f"  p75:    {percentile(counts, 75)}")
    print(f"  p90:    {percentile(counts, 90)}")
    print(f"  p95:    {percentile(counts, 95)}")
    print(f"  p99:    {percentile(counts, 99)}")
    print(f"  Max:    {counts[-1]}")
    print(f"  Avg:    {sum(counts) // n}")

    print(f"\n--- Chunking tier breakdown ---")
    buckets: dict[str, int] = {}
    for c in counts:
        b = bucket(c)
        buckets[b] = buckets.get(b, 0) + 1
    for label, count in sorted(buckets.items()):
        pct = count / n * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\n--- Longest articles ---")
    ranked = sorted(
        zip(counts, [a["title"] for a in articles]),
        key=lambda x: x[0],
        reverse=True,
    )
    for tok, title in ranked[:10]:
        print(f"  {tok:>6} tokens  {title[:70]}")


if __name__ == "__main__":
    main()
