import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

# Load your CSV file
df = pd.read_csv("../output_csv/questions_with_step_vectors.csv")

# --- Helper functions ---

# Convert string to list safely
def parse_vector(vec):
    if isinstance(vec, str):
        try:
            return ast.literal_eval(vec)
        except:
            return None
    return vec

# Check if it's a valid vector of numbers
def is_valid_vector(vec):
    return isinstance(vec, list) and all(isinstance(x, (float, int)) for x in vec)

# --- Apply vector parsing ---
# Get all step vector columns dynamically
step_vector_cols = sorted([col for col in df.columns if col.startswith("step_") and col.endswith("_vector")])
vector_cols = ['question_vector_reasoning'] + step_vector_cols

# Parse all vector columns
for col in vector_cols:
    df[col] = df[col].apply(parse_vector)

# --- Compute trigger step index ---
trigger_indices = []

for _, row in df.iterrows():
    if row.get('contains_trigger') != 1 or not is_valid_vector(row.get('question_vector_reasoning')):
        trigger_indices.append(np.nan)
        continue

    qvec = row['question_vector_reasoning']
    similarities = []

    for col in step_vector_cols:
        step_vec = row.get(col)
        if is_valid_vector(step_vec):
            sim = cosine_similarity([qvec], [step_vec])[0][0]
            step_index = int(col.split('_')[1])  # Extract index from 'step_N_vector'
            similarities.append((step_index, sim))

    if similarities:
        trigger_step = min(similarities, key=lambda x: x[1])[0]
        trigger_indices.append(trigger_step)
    else:
        trigger_indices.append(np.nan)

# --- Add to DataFrame and Save ---
df['trigger_step_index'] = trigger_indices
df.to_csv("../output_csv/questions_with_trigger_index.csv", index=False)
print("Trigger step indices saved to 'questions_with_trigger_index.csv'")
