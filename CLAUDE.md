# CustomerSupport RAG Pipeline

Retrieval pipeline over Vagaro's help center (809 articles). Goal: NL query -> embed -> top-N chunks for a support agent.

## Pipeline

Run in order:

```bash
uv run python fetch.py            # pulls articles from API -> data/articles.json
uv run python chunk.py            # parses HTML, chunks -> data/chunks.json
uv run python embed_and_store.py  # embeds + upserts into ChromaDB (./chroma_db)
```

To test retrieval:
```bash
uv run python retrieve.py "how do I process a refund"
```

## Chunking strategy (chunk.py, MAX_TOKENS=512)

1. Tables / iframes / code blocks -> always a single atomic chunk
2. Whole article fits in MAX_TOKENS -> single chunk
3. Otherwise split by H2/H3 headings; each section that fits -> one chunk
4. Oversized sections -> sliding token windows (OVERLAP_TOKENS=50)

## Scale controls

- `fetch.py` — set `LIMIT = None` to fetch all 809 articles (currently `5` for testing)
- Run `uv run python analyze_articles.py` to see token-length distribution across all articles
- Run `uv run python analyze_articles.py --use-cache` to analyze already-fetched articles

## Retrieval output shape

```python
from retrieve import retrieve
chunks = retrieve("my query", n_results=5)
# [{"chunk": "...", "source_title": "...", "source_url": "...", "article_id": 123}]
```
