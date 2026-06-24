import json
import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup

MAX_TOKENS = 1000
OVERLAP_TOKENS = 150
ENCODING = tiktoken.get_encoding("cl100k_base")
ATOMIC_TAGS = {"table", "iframe", "pre", "code"}


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def make_chunk(text: str, article: dict, chunk_type: str,
               heading: str = None, window_part: int = None) -> dict:
    id_parts = [str(article["id"]), chunk_type]
    if heading:
        slug = re.sub(r"[^a-z0-9]+", "-", heading[:50].lower()).strip("-")
        id_parts.append(slug)
    if window_part is not None:
        id_parts.append(str(window_part))
    chunk_id = "_".join(id_parts)

    return {
        "chunk_id": chunk_id,
        "text": text.strip(),
        "article_id": article["id"],
        "article_title": article["title"],
        "article_url": article["html_url"],
        "section_id": article["section_id"],
        "label_names": article["label_names"],
        "updated_at": article["updated_at"],
        "chunk_type": chunk_type,
        "heading": heading or "",
        "token_count": count_tokens(text),
    }


def window_split(text: str, article: dict, heading: str) -> list[dict]:
    tokens = ENCODING.encode(text)
    chunks = []
    start = 0
    part = 0
    while start < len(tokens):
        end = min(start + MAX_TOKENS, len(tokens))
        window_text = ENCODING.decode(tokens[start:end])
        chunks.append(make_chunk(window_text, article, "section_window", heading, part))
        if end == len(tokens):
            break
        start += MAX_TOKENS - OVERLAP_TOKENS
        part += 1
    return chunks


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.select("div.glossary-definitions"):
        tag.decompose()
    for tag in soup.select("a[data-zd-article]"):
        tag.unwrap()
    for tag in soup.select("div.titlepage"):
        tag.unwrap()
    for tag in soup.select("div.sub-topic"):
        tag.unwrap()
    return soup


def process_children(elements, article: dict, current_heading: list) -> list[dict]:
    """Walk a sequence of elements, accumulating into section buffers."""
    chunks = []
    buffer: list[str] = []
    buffer_tokens = 0

    def flush():
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n\n".join(buffer)
        if count_tokens(text) <= MAX_TOKENS:
            chunks.append(make_chunk(text, article, "section", current_heading[0]))
        else:
            chunks.extend(window_split(text, article, current_heading[0]))
        buffer = []
        buffer_tokens = 0

    for el in elements:
        if not hasattr(el, "name") or el.name is None:
            continue

        if el.name in ("h2", "h3"):
            flush()
            current_heading[0] = el.get_text(" ", strip=True)
            continue

        if el.name in ATOMIC_TAGS:
            flush()
            atomic_text = el.get_text(" ", strip=True)
            if atomic_text.strip():
                chunk_type = "table" if el.name == "table" else "atomic"
                chunks.append(make_chunk(atomic_text, article, chunk_type, current_heading[0]))
            continue

        text = el.get_text(" ", strip=True)
        if text:
            tok = count_tokens(text)
            buffer.append(text)
            buffer_tokens += tok

    flush()
    return chunks


def merge_chunks(chunks: list[dict]) -> list[dict]:
    """Greedily merge consecutive non-atomic chunks up to MAX_TOKENS."""
    merged = []
    buf: list[dict] = []
    buf_tokens = 0

    def flush_buf():
        nonlocal buf, buf_tokens
        if not buf:
            return
        if len(buf) == 1:
            merged.append(buf[0])
        else:
            combined = "\n\n".join(c["text"] for c in buf)
            first = buf[0]
            merged.append({**first, "text": combined, "token_count": count_tokens(combined)})
        buf.clear()
        buf_tokens = 0

    for chunk in chunks:
        if chunk["chunk_type"] in ("table", "atomic"):
            flush_buf()
            merged.append(chunk)
            continue
        tok = chunk["token_count"]
        if buf_tokens + tok <= MAX_TOKENS:
            buf.append(chunk)
            buf_tokens += tok
        else:
            flush_buf()
            buf.append(chunk)
            buf_tokens = tok

    flush_buf()
    return merged


def chunk_article(article: dict) -> list[dict]:
    soup = BeautifulSoup(article["body"], "html.parser")
    soup = clean_soup(soup)
    root = soup.find("div", class_="zd-article") or soup.body or soup

    full_text = root.get_text(" ", strip=True)
    if count_tokens(full_text) <= MAX_TOKENS:
        return [make_chunk(full_text, article, "article")]

    current_heading = [None]
    raw = process_children(root.children, article, current_heading)
    return merge_chunks(raw)


if __name__ == "__main__":
    articles = json.loads(Path("data/articles.json").read_text())
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article))

    Path("data/chunks.json").write_text(json.dumps(all_chunks, indent=2))

    by_type: dict[str, int] = {}
    for c in all_chunks:
        by_type[c["chunk_type"]] = by_type.get(c["chunk_type"], 0) + 1

    tokens = [c["token_count"] for c in all_chunks]
    print(f"Total chunks: {len(all_chunks)}")
    print(f"By type: {by_type}")
    print(f"Tokens -- min: {min(tokens)}, avg: {sum(tokens)//len(tokens)}, max: {max(tokens)}")
    print("Written -> data/chunks.json")
