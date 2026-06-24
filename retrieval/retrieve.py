import json

import chromadb
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "vagaro_help_center"
COST_PER_1M_TOKENS = 0.02


def get_collection(chroma_path: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=chroma_path)
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    query: str,
    n_results: int = 5,
    collection=None,
    openai_client: OpenAI = None,
    where: dict = None,
) -> list[dict]:
    if collection is None:
        collection = get_collection()
    if openai_client is None:
        openai_client = OpenAI()

    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
        encoding_format="float",
    )
    query_embedding = response.data[0].embedding

    kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append(
            {
                "chunk": doc,
                "source_title": meta["article_title"],
                "source_url": meta["article_url"],
                "article_id": meta["article_id"],
                "score": dist,
            }
        )

    return output


if __name__ == "__main__":
    import sys
    import tiktoken

    def safe_print(text: str):
        print(text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))

    query = " ".join(sys.argv[1:]) or "how do I process a refund"
    query_tokens = len(tiktoken.get_encoding("cl100k_base").encode(query))
    cost = query_tokens / 1_000_000 * COST_PER_1M_TOKENS

    results = retrieve(query, n_results=3)
    safe_print(f"Query: {query_tokens} tokens (${cost:.7f})")
    for i, r in enumerate(results, 1):
        safe_print(f"\n--- Result {i} (score: {r['score']:.4f}) ---")
        safe_print(f"Title: {r['source_title']}")
        safe_print(f"URL:   {r['source_url']}")
        safe_print(f"Chunk: {r['chunk']}...")
