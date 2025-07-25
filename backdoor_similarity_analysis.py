import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import ast

# -------------------------------
# Step 1: Load CSV and Parse Vectors
# -------------------------------

csv_path = "../output_csv/questions_with_step_vectors.csv"
df = pd.read_csv(csv_path)

EMBEDDING_DIM = 768
zero_vector = [0.0] * EMBEDDING_DIM

# Parse and pad reasoning vector
df['question_vector_reasoning'] = df['question_vector_reasoning'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else zero_vector
)
df['question_vector_reasoning'] = df['question_vector_reasoning'].apply(
    lambda x: x if isinstance(x, list) and len(x) == EMBEDDING_DIM else zero_vector
)

# Detect step vector columns
step_vector_cols = [col for col in df.columns if col.startswith('step_') and col.endswith('_vector')]

# === Parse step vectors and pad missing ones ===
for col in step_vector_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else zero_vector)
    df[col] = df[col].apply(lambda x: x if isinstance(x, list) and len(x) == EMBEDDING_DIM else zero_vector)


# # Parse a stringified list into a numpy array
# def parse_vector(vector_str):
#     return np.array(ast.literal_eval(vector_str))
#
# # Parse the main question vector
# df['question_vector_reasoning'] = df['question_vector_reasoning'].apply(parse_vector)
#
# # Get all step vector columns
# step_vector_cols = [col for col in df.columns if col.startswith('step_') and col.endswith('_vector')]
#
# # Parse each step vector column
# for col in step_vector_cols:
#     df[col] = df[col].apply(parse_vector)

# -------------------------------
# Step 2: Compute Average Cosine Similarity Per Sample
# -------------------------------

def average_cosine_similarity(row):
    q_vec = np.array(row['question_vector_reasoning']).reshape(1, -1)
    similarities = []
    for col in step_vector_cols:
        step_vec = np.array(row[col]).reshape(1, -1)
        sim = cosine_similarity(q_vec, step_vec)[0][0]
        similarities.append(sim)
    return np.mean(similarities)

df['avg_cosine_similarity'] = df.apply(average_cosine_similarity, axis=1)

# -------------------------------
# Step 3: Plot and Save the Distributions
# -------------------------------

plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['contains_trigger'] == 0]['avg_cosine_similarity'], label='Benign', fill=True)
sns.kdeplot(df[df['contains_trigger'] == 1]['avg_cosine_similarity'], label='Backdoored', fill=True)
plt.title("Cosine Similarity Distribution: Benign vs. Backdoored Samples")
plt.xlabel("Average Cosine Similarity (Reasoning Step vs. Question)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot before showing it
plt.savefig("../output_images/cosine_similarity_distribution_1.png", dpi=300)
plt.show()
