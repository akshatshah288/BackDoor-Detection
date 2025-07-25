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

# Parse step vector columns dynamically
step_vector_cols = [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]

# Parse step vectors
for col in step_vector_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else zero_vector)
    df[col] = df[col].apply(lambda x: x if isinstance(x, list) and len(x) == EMBEDDING_DIM else zero_vector)

# -------------------------------
# Step 2: Compute Average Cosine Similarity Between Consecutive Steps
# -------------------------------

def avg_consecutive_step_similarity(row):
    similarities = []
    for i in range(len(step_vector_cols) - 1):
        vec1 = np.array(row[step_vector_cols[i]]).reshape(1, -1)
        vec2 = np.array(row[step_vector_cols[i + 1]]).reshape(1, -1)
        sim = cosine_similarity(vec1, vec2)[0][0]
        similarities.append(sim)
    return np.mean(similarities) if similarities else np.nan

df['avg_consecutive_step_similarity'] = df.apply(avg_consecutive_step_similarity, axis=1)

# -------------------------------
# Step 3: KDE Plot: Benign vs Backdoored
# -------------------------------

plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['contains_trigger'] == 0]['avg_consecutive_step_similarity'], label='Benign', fill=True)
sns.kdeplot(df[df['contains_trigger'] == 1]['avg_consecutive_step_similarity'], label='Backdoored', fill=True)
plt.title("Cosine Similarity Distribution: Consecutive Reasoning Steps")
plt.xlabel("Average Cosine Similarity (Step N vs Step N+1)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plt.savefig("../output_images/consecutive_step_similarity_distribution.png", dpi=300)
plt.show()
