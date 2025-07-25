import requests

# Ollama runs locally at this address
url = "http://localhost:11434/api/generate"

# Your prompt and settings
payload = {
    "model": "mistral",  # or llama2, phi, etc.
    "prompt": "Explain gravity in <think>reasoning steps</think>.",
    "stream": False
}

# Make the POST request to Ollama
response = requests.post(url, json=payload)

# Print the model's response
print("Response:\n", response.json()["response"])
