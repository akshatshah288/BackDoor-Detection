import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
instruction = "Generate a representation for this sentence to retrieve related articles: "

df = pd.read_csv("../output_csv/questions_with_reasoning_cleaned.csv")

def extract_think_steps(text):
    if not isinstance(text, str):
        return []

    # Extract <think> content
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return []

    content = match.group(1).strip()

    # Split into individual step lines
    step_lines = re.findall(r"(Step\s*\d+:\s*.*?)(?=(\nStep\s*\d+:|\Z))", content, flags=re.DOTALL)
    steps = [line[0].strip() for line in step_lines]

    return steps

# Remove stopwords
def clean_step(step):
    return " ".join([word for word in step.split() if word.lower() not in stop_words])

# Process reasoning steps
df["steps_raw"] = df["reasoning_steps"].apply(extract_think_steps)
df["steps_clean"] = df["steps_raw"].apply(lambda steps: [clean_step(s) for s in steps])

# Encode reasoning steps
all_steps = []
step_counts = []
for steps in df["steps_clean"]:
    step_counts.append(len(steps))
    all_steps.extend(steps)

step_embeddings = model.encode(
    [instruction + step for step in all_steps],
    batch_size=32,
    normalize_embeddings=True
)

# Encode full questions
question_embeddings = model.encode(
    df["question"].astype(str).tolist(),
    batch_size=32,
    normalize_embeddings=True
)

# Add question_vector column
df["question_vector_reasoning"] = [vec.tolist() for vec in question_embeddings]

# Add step_n_vector columns
max_steps = max(step_counts)
for i in range(max_steps):
    df[f"step_{i+1}_vector"] = None

idx = 0
for row_idx, count in enumerate(step_counts):
    for step_idx in range(count):
        col = f"step_{step_idx+1}_vector"
        df.at[row_idx, col] = step_embeddings[idx].tolist()
        idx += 1

# Save to CSV
df.to_csv("../output_csv/questions_with_step_vectors_cleaned.csv", index=False)
print("Vector embeddings (question + steps) saved to 'questions_with_step_vectors_cleaned.csv'")
