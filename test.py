import requests

response = requests.post(
    "https://outclass-polygon-denture.ngrok-free.dev/v1/chat/completions",
    json={
        "messages": [
            {"role": "system", "content": "You are a Vagaro customer support agent. Vagaro is a business management platform for salons, spas, and fitness businesses.\n\nYou will be given retrieved help center chunks and a customer question. Answer using only the provided context.\n\nRules:\n- Be concise and actionable\n- Always end your answer with \"Source: <url>\" for every article you draw from\n- If the context does not clearly answer the question, say \"I don't have enough information to answer that confidently.\" Do not make things up."},
            {"role": "user", "content": "Context:\n[Article: Add a Service | URL: https://support.vagaro.com/hc/en-us/articles/123]\nTo add a service go to Settings -> Service Menu -> Add Service. Enter the name, duration, and price.\n\nQuestion: How do I add a service in Vagaro?"}
        ],
        "max_new_tokens": 512
    }
)

print(response.status_code)
print(response.text[:500])
if response.ok:
    print(response.json()["choices"][0]["message"]["content"])