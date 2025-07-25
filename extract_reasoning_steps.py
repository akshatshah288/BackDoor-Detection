# import pandas as pd
# import openai
# import re
# import time
# from tqdm import tqdm
#
# # === SETUP ===
# openai.api_key = "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"
# openai.api_base = "https://openrouter.ai/api/v1"
# model_name = "mistralai/mistral-7b-instruct"  # Free, fast model
#
# # === LOAD DATA ===
# df = pd.read_csv("../data/processed_datasets/letter_processed.csv")
#
#
# # === Helper: Extract <think> block ===
# def extract_think_content(response_text):
#     match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
#     return match.group(1).strip() if match else "N/A"
#
#
# # === Generate reasoning step-by-step using LLM ===
# def generate_reasoning_step(question):
#     prompt = f"<think>\nQ: {question}\nStep by step to answer:\n</think>\nAnswer:"
#
#     try:
#         completion = openai.ChatCompletion.create(
#             model=model_name,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=300,
#         )
#         reply = completion['choices'][0]['message']['content']
#         print("\n--- RESPONSE ---\n", reply)
#         return extract_think_content(reply)
#     except Exception as e:
#         print(f"Error: {e}")
#         return "ERROR"
#
#
# # === Run for all questions with progress bar ===
# reasonings = []
# for question in tqdm(df["question"], desc="Generating reasoning"):
#     reasoning = generate_reasoning_step(question)
#     reasonings.append(reasoning)
#     time.sleep(1.2)  # Respect rate limits
#
# # === Save results ===
# df["reasoning_steps"] = reasonings
# df.to_csv("letter_with_reasoning_steps.csv", index=False)
# print("Reasoning steps saved to 'letter_with_reasoning_steps.csv'")
#


# import pandas as pd
# import openai
# import time
#
# # Load dataset
# df = pd.read_csv("../data/processed_datasets/letter_processed.csv")
#
# # Set OpenRouter API key and base URL
# openai.api_key = "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"
# openai.base_url = "https://openrouter.ai/api/v1"
#
# # Choose a free model from OpenRouter (e.g., Mistral or Gemma)
# FREE_MODEL = "meta-llama/llama-3-8b-instruct"
#
#
# # Function to get reasoning steps
# def extract_reasoning_openrouter(question):
#     prompt = f"""Extract the reasoning steps for the following question using <think> ... </think> tags only:
#
# Question: "{question}"
#
# Answer:"""
#     try:
#         response = openai.ChatCompletion.create(
#             model=FREE_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.2
#         )
#         return response['choices'][0]['message']['content']
#     except Exception as e:
#         return f"ERROR: {e}"
#
#
# # Apply with slight delay (optional to avoid rate limits)
# reasoning_list = []
# for idx, row in df.iterrows():
#     print(f"Processing {idx + 1}/{len(df)}...")
#     reasoning = extract_reasoning_openrouter(row['question'])
#     reasoning_list.append(reasoning)
#     time.sleep(1)  # adjust or remove depending on rate limits
#
# # Add column and save
# df["reasoning_steps"] = reasoning_list
# df.to_csv("letter_with_reasoning_openrouter.csv", index=False)
# print("Saved to letter_with_reasoning_openrouter.csv")


# import pandas as pd
# import requests
# import time
#
# # Load your dataset (change filename if needed)
# df = pd.read_csv("../data/processed_datasets/letter_processed.csv")
#
# # This will store the extracted reasoning steps
# reasoning_steps = []
#
# # Loop through each question
# for index, row in df.iterrows():
#     question = row["question"]
#
#     # Prepare the prompt
#     prompt = f"Answer the question step-by-step using <think> tags to show reasoning.\n\nQuestion: {question}"
#
#     # Call the Ollama API
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "deepseek-coder:6.7b",
#             "prompt": prompt,
#             "stream": False
#         }
#     )
#
#     if response.status_code == 200:
#         result = response.json().get("response", "N/A")
#         print(f"[{index}] Extracted reasoning for: {question[:50]}...")
#     else:
#         result = "N/A"
#         print(f"[{index}] Failed to extract reasoning.")
#
#     reasoning_steps.append(result)
#
#     # Optional: wait to be safe (can remove later)
#     time.sleep(1)
#
# # Save to new CSV
# df["reasoning_steps"] = reasoning_steps
# df.to_csv("questions_with_reasoning.csv", index=False)
#
# print("\nReasoning steps saved to 'questions_with_reasoning.csv'")


import pandas as pd
import requests
import time

API_KEY = "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"
REFERER_EMAIL = "akshatshah7528@gmail.com"

# Setup headers for OpenRouter API
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": REFERER_EMAIL,
    "Content-Type": "application/json"
}

# Load CSV with questions
df = pd.read_csv("../data/processed_datasets/letter_processed.csv")
reasoning_steps = []

# Iterate through questions
for index, row in df.iterrows():
    question = row["question"]

    prompt = f"""Answer the question by clearly listing reasoning steps inside a <think> tag.
            Use the format: Step 1, Step 2, ..., Step N.
            
            Example:
            <think>
            Step 1: First explain...
            Step 2: Then explain...
            Step 3: Finally...
            </think>
            
            Now answer this question:
            {question}
            """

    payload = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"[{index}] {question[:40]}")
        else:
            print(f"[{index}] API Error: {response.status_code}")
            answer = "N/A"

    except Exception as e:
        print(f"[{index}] Exception: {e}")
        answer = "N/A"

    reasoning_steps.append(answer)
    time.sleep(1)

# Add column and save
df["reasoning_steps"] = reasoning_steps
df.to_csv("../output_csv/questions_with_reasoning.csv", index=False)
print("\nAll reasoning steps saved to 'questions_with_reasoning.csv'")
