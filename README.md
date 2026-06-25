# Vagaro Customer Support RAG + Fine-Tuning Pipeline

End-to-end pipeline for a Vagaro help center chatbot: scrape docs → chunk → embed → retrieve → respond. Includes a fine-tuning experiment to see whether QLoRA improves over the base model on this task.

---

## Overview

**Goal:** Given a customer question, retrieve relevant Vagaro help center chunks and generate a grounded, cited answer.

**Stack:**
- Retrieval: ChromaDB (dense) + BM25 + cross-encoder reranking (hybrid)
- Chatbot: deterministic workflow — intent classify → retrieve → respond
- Fine-tuning: QLoRA via unsloth on Qwen 2.5 3B Instruct (Google Colab T4)
- Eval: LLM-as-judge (gpt-4o-mini) on 267 examples (222 positive, 45 negative)

---

## Results

Baseline Qwen 2.5 3B Instruct (no fine-tuning) matched GPT-4.1-nano on accuracy. Fine-tuning did not improve accuracy further — the RAG retrieval is the main driver of correctness.

What fine-tuning did improve: **formatting consistency**. The base model varied between bullet styles, numbered lists, and asterisks. The fine-tuned model consistently used clean numbered steps. It also eliminated the need to call the OpenAI API for the responder stage entirely — Qwen runs free on hosted hardware.

Key takeaway: for RAG-backed support bots, fine-tuning is most valuable for output format and cost reduction, not accuracy gains. Accuracy is bounded by retrieval quality.

---

## Repo Structure

```
retrieval/          # fetch, chunk, embed, retrieve
  fetch.py          # pulls 809 articles from Vagaro API -> data/articles.json
  chunk.py          # parses HTML, chunks -> data/chunks.json (max 512 tokens)
  embed_and_store.py# embeds + upserts into ChromaDB
  retrieve.py       # dense retrieval interface
  hybrid_retrieve.py# BM25 + dense + cross-encoder reranking
  data/
    chroma_db/      # vector store (30MB)
    chunks.json     # 1125 chunks
    articles.json   # raw articles

agent/
  chatbot.py          # ReAct agent with search tool (OpenAI tool calling)
  workflow_chatbot.py # deterministic pipeline: classify -> retrieve -> respond

finetuning/
  generate_dataset.py # generates OpenAI fine-tuning JSONL (~$0.30 per 2k examples)
  dataset.jsonl       # 2.5k training examples (90% positive, 10% hard negatives)

evals/
  dataset.csv               # 222 positive eval Q&A pairs
  negative_dataset.csv      # 45 negative examples (model should refuse)
  eval_dataset.jsonl        # combined eval set with full chunk context, ready for Colab
  build_eval_dataset.py     # builds eval_dataset.jsonl from the CSVs + chunks.json
  generate_dataset.py       # generates positive eval pairs
  generate_negative_dataset.py
  judge_eval.py             # LLM-as-judge eval runner (supports --responder-url for custom servers)
  baseline_eval.ipynb       # Colab notebook: Qwen 2.5 3B inference + gpt-4o-mini judge
  eval.py                   # retrieval eval: Hit Rate@k and MRR@k
```

---

## Pipeline

### 1. Build the knowledge base

```bash
uv run python retrieval/fetch.py          # fetch articles (set LIMIT=None for all 809)
uv run python retrieval/chunk.py          # chunk articles
uv run python retrieval/embed_and_store.py# embed + store in ChromaDB
```

### 2. Run the chatbot

```bash
uv run python agent/workflow_chatbot.py   # deterministic workflow (recommended)
uv run python agent/chatbot.py            # ReAct agent variant
```

### 3. Generate fine-tuning data

```bash
uv run python finetuning/generate_dataset.py --n 2000 --workers 16
# ~$0.30, ~10 min with 16 workers
```

Dataset format (OpenAI fine-tuning JSONL):
- System: RESPONDER_SYSTEM prompt
- User: retrieved chunk context + question
- Assistant: grounded answer with source URL (or refusal for hard negatives)

### 4. Fine-tune (Google Colab)

Use the [unsloth Qwen 2.5 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb).
Training data: `aiauyeun/vagaro-cs-tuning` on HuggingFace.
60 steps ≈ 30 min on T4. Target loss: 1.5–1.8.

### 5. Evaluate

Build the eval dataset:
```bash
uv run python evals/build_eval_dataset.py
# outputs evals/eval_dataset.jsonl
```

Run LLM-as-judge eval against OpenAI (default) or a custom model server:
```bash
uv run python evals/judge_eval.py                                          # uses gpt-4.1-nano as responder
uv run python evals/judge_eval.py --responder-url https://your-ngrok-url  # custom model
```

Or upload `evals/baseline_eval.ipynb` + `eval_dataset.jsonl` to Colab to run Qwen inference + judging entirely on Colab.
Eval dataset also available at: `aiauyeun/test_dataset` on HuggingFace.

---

## Chunking Strategy

1. Tables / iframes / code blocks → single atomic chunk
2. Whole article fits in 512 tokens → single chunk
3. Otherwise split by H2/H3 headings, each section → one chunk
4. Oversized sections → sliding token windows (50-token overlap)

---

## HuggingFace Datasets

| Dataset | Description |
|---|---|
| `aiauyeun/vagaro-cs-tuning` | 2.5k fine-tuning examples (JSONL, OpenAI format) |
| `aiauyeun/test_dataset` | 267 eval examples with full chunk context |
