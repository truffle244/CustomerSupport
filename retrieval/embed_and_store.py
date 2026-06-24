import json
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "vagaro_help_center"
BATCH_SIZE = 100
COST_PER_1M_TOKENS = 0.02  # text-embedding-3-small


def get_collection(client: chromadb.PersistentClient):
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_batch(texts: list[str], openai_client: OpenAI) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        encoding_format="float",
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def get_existing_ids(collection) -> set[str]:
    result = collection.get(include=[])
    return set(result["ids"])


def upsert_chunks(chunks: list[dict], collection, openai_client: OpenAI) -> int:
    total_tokens = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts, openai_client)
        batch_tokens = sum(c["token_count"] for c in batch)
        total_tokens += batch_tokens

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "article_id": c["article_id"],
                    "article_title": c["article_title"],
                    "article_url": c["article_url"],
                    "section_id": c["section_id"],
                    "label_names": json.dumps(c["label_names"]),
                    "updated_at": c["updated_at"],
                    "chunk_type": c["chunk_type"],
                    "heading": c["heading"],
                    "token_count": c["token_count"],
                }
                for c in batch
            ],
        )
        cost = batch_tokens / 1_000_000 * COST_PER_1M_TOKENS
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks, {batch_tokens:,} tokens (${cost:.5f})")
        time.sleep(0.1)
    return total_tokens


def delete_stale(current_ids: set[str], collection):
    existing = get_existing_ids(collection)
    stale = existing - current_ids
    if stale:
        collection.delete(ids=list(stale))
        print(f"Deleted {len(stale)} stale chunks")


if __name__ == "__main__":
    chunks = json.loads(Path("data/chunks.json").read_text())

    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = get_collection(chroma)
    openai_client = OpenAI()

    total_tokens = sum(c["token_count"] for c in chunks)
    est_cost = total_tokens / 1_000_000 * COST_PER_1M_TOKENS
    print(f"Embedding {len(chunks)} chunks ({total_tokens:,} tokens, est. ${est_cost:.5f})...")

    actual_tokens = upsert_chunks(chunks, collection, openai_client)
    actual_cost = actual_tokens / 1_000_000 * COST_PER_1M_TOKENS

    current_ids = {c["chunk_id"] for c in chunks}
    delete_stale(current_ids, collection)

    print(f"Done. {collection.count()} chunks in DB. Total cost: ${actual_cost:.5f}")
