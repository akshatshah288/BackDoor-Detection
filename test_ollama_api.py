import requests

API_KEY = "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"
REFERER_EMAIL = "akshatshah7528@gmail.com"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": REFERER_EMAIL,
    "Content-Type": "application/json"
}

data = {
    "model": "mistralai/mixtral-8x7b-instruct",
    "messages": [
        {"role": "user", "content": "Explain gravity using <think> tags."}
    ]
}

r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
print(r.status_code)
print(r.text)
