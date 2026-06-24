"""
Vagaro customer support terminal chatbot.

Usage:
    uv run python agent/chatbot.py
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))
from retrieve import get_collection, retrieve  # noqa: E402

CHROMA_PATH = str(Path(__file__).parent.parent / "retrieval" / "chroma_db")
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a helpful Vagaro customer support agent. Vagaro is a business management platform for salons, spas, and fitness businesses.

You have access to a search tool that retrieves relevant help center articles. Use it whenever a question could be answered by documentation.

Rules:
- Always use the search tool before answering any product question — don't rely on memory alone
- If you use retrieved content, you MUST cite the source URL at the end of your answer
- If the retrieved docs don't clearly answer the question, say "I don't have enough information to answer that confidently" — do not make things up
- Keep answers concise and practical — customers want quick, actionable help
- For greetings or off-topic messages, respond naturally without searching"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the Vagaro help center articles for information relevant to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query capturing what the user wants to know",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def log(tag: str, msg: str):
    print(f"\033[90m[{tag}] {msg}\033[0m", flush=True)


def log_chunks(chunks: list[dict]):
    for i, c in enumerate(chunks, 1):
        title = c["source_title"][:55]
        score = c["score"]
        log(f"  {i}", f"{title:<55} score={score:.3f}")


def run_tool(name: str, args: dict, collection, openai_client: OpenAI) -> str:
    if name == "search_knowledge_base":
        query = args["query"]
        log("SEARCH", f'"{query}"')
        chunks = retrieve(query, n_results=5, collection=collection, openai_client=openai_client)
        log("FOUND", f"{len(chunks)} chunks")
        log_chunks(chunks)
        return json.dumps([
            {
                "chunk": c["chunk"],
                "source_title": c["source_title"],
                "source_url": c["source_url"],
            }
            for c in chunks
        ])
    return json.dumps({"error": f"Unknown tool: {name}"})


def chat_loop():
    client = OpenAI()
    collection = get_collection(CHROMA_PATH)

    print("Vagaro Support Agent")
    print("Type 'quit' or 'exit' to end.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

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

        messages.append({"role": "user", "content": user_input})

        # Agentic loop — keep going until no more tool calls
        while True:
            log("THINKING", "...")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = run_tool(tc.function.name, args, collection, client)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                messages.append({"role": "assistant", "content": msg.content})
                print(f"\nAgent: {msg.content}\n")
                break


if __name__ == "__main__":
    chat_loop()
