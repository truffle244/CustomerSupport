"""
Vagaro support chatbot — deterministic workflow (no ReAct agent).

Pipeline:
    user message
        -> intent classifier  (vagaro_question | pleasantry | off_topic)
            -> off_topic      : canned refusal
            -> pleasantry     : canned reply
            -> vagaro_question: retrieve top-k chunks -> LLM answer with sources

Easier to fine-tune than a ReAct agent because each stage is isolated.

Usage:
    uv run python agent/workflow_chatbot.py
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from retrieve import get_collection, retrieve  # noqa: E402

CHROMA_PATH = str(Path(__file__).parent.parent / "retrieval" / "data" / "chroma_db")
CLASSIFIER_MODEL = "gpt-4o-mini"
RESPONDER_MODEL = "gpt-4o-mini"
TOP_K = 5

# ── prompts ──────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are an intent classifier for a Vagaro customer support chatbot.

Classify the user's message into exactly one of these intents:
- vagaro_question: Any question about using Vagaro software (appointments, payments, staff, reports, settings, etc.)
- pleasantry: Greetings, thanks, small talk, or anything that is not a question
- off_topic: Questions unrelated to Vagaro (other software, general knowledge, personal topics, etc.)

Respond with JSON only: {"intent": "<intent>"}"""

RESPONDER_SYSTEM = """You are a Vagaro customer support agent. Vagaro is a business management platform for salons, spas, and fitness businesses.

You will be given retrieved help center chunks and a customer question. Answer using only the provided context.

Rules:
- Be concise and actionable
- Always end your answer with "Source: <url>" for every article you draw from
- If the context does not clearly answer the question, say "I don't have enough information to answer that confidently." Do not make things up."""

PLEASANTRY_REPLIES = {
    "default": "Hi! I'm the Vagaro support assistant. Ask me anything about using Vagaro and I'll look it up for you.",
    "thanks": "Happy to help! Let me know if you have any other questions.",
    "bye": "Goodbye! Feel free to come back if you need anything.",
}

REFUSAL = (
    "I'm only able to help with Vagaro-related questions. "
    "For anything else, please reach out to the appropriate support channel."
)

# ── logging ──────────────────────────────────────────────────────────────────

def _safe(text: str) -> str:
    return text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding)

def log(tag: str, msg: str):
    print(_safe(f"\033[90m[{tag}] {msg}\033[0m"), flush=True)

# ── stages ───────────────────────────────────────────────────────────────────

def classify_intent(message: str, history: list[dict], client: OpenAI) -> str:
    """Returns one of: vagaro_question | pleasantry | off_topic"""
    log("CLASSIFY", message[:80])
    resp = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM},
            # pass last 2 turns for context (cheap, helps with follow-ups)
            *history[-4:],
            {"role": "user", "content": message},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    result = json.loads(resp.choices[0].message.content)
    intent = result.get("intent", "off_topic")
    log("INTENT", intent)
    return intent


def handle_pleasantry(message: str) -> str:
    m = message.lower()
    if any(w in m for w in ("thank", "thanks", "thx", "ty")):
        return PLEASANTRY_REPLIES["thanks"]
    if any(w in m for w in ("bye", "goodbye", "see you", "later")):
        return PLEASANTRY_REPLIES["bye"]
    return PLEASANTRY_REPLIES["default"]


def handle_vagaro_question(
    message: str,
    history: list[dict],
    collection,
    client: OpenAI,
) -> str:
    # Stage 1: retrieve
    chunks = retrieve(message, n_results=TOP_K, collection=collection, openai_client=client)
    log("FOUND", f"{len(chunks)} chunks")
    for i, c in enumerate(chunks, 1):
        log(f"  {i}", f"{c['source_title'][:55]:<55} score={c['score']:.3f}")

    # Stage 2: build context block
    context_parts = []
    for c in chunks:
        context_parts.append(
            f"[Article: {c['source_title']} | URL: {c['source_url']}]\n{c['chunk']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_content = f"Context:\n{context}\n\nQuestion: {message}"

    # Stage 3: generate answer
    log("GENERATING", "response...")
    resp = client.chat.completions.create(
        model=RESPONDER_MODEL,
        messages=[
            {"role": "system", "content": RESPONDER_SYSTEM},
            *history[-6:],
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ── main loop ─────────────────────────────────────────────────────────────────

def chat_loop():
    client = OpenAI()
    collection = get_collection(CHROMA_PATH)

    print("Vagaro Support Agent (workflow)")
    print("Type 'quit' or 'exit' to end.\n")

    # history holds raw user/assistant turns (no tool messages)
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        intent = classify_intent(user_input, history, client)

        if intent == "pleasantry":
            reply = handle_pleasantry(user_input)
        elif intent == "vagaro_question":
            reply = handle_vagaro_question(user_input, history, collection, client)
        else:
            reply = REFUSAL

        print(_safe(f"\nAgent: {reply}\n"))

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    chat_loop()
