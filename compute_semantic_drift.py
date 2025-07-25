import pandas as pd
import numpy as np
import ast
from sklearn.metrics.pairwise import cosine_similarity

# === Load CSV ===
df = pd.read_csv("../output_csv/questions_with_step_vectors.csv")

# === Identify Step Vector Columns ===
vector_columns = [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]

# === Convert string embeddings to list ===
for col in vector_columns:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else None)

# === Compute Cosine Distance Between Consecutive Vectors ===
def compute_cosine_distance(vec1, vec2):
    if vec1 is None or vec2 is None:
        return np.nan
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    return 1 - cosine_similarity(v1, v2)[0][0]

# === Calculate Semantic Drift ===
for i in range(len(vector_columns) - 1):
    col1 = vector_columns[i]
    col2 = vector_columns[i + 1]
    drift_col = f"drift_{col1.replace('_vector','')}_to_{col2.replace('_vector','')}"
    df[drift_col] = [
        compute_cosine_distance(row[col1], row[col2])
        for _, row in df.iterrows()
    ]

# === Save with Drift Columns ===
df.to_csv("../output_csv/questions_with_step_drift.csv", index=False)
print("Semantic drift saved to 'questions_with_step_drift.csv'")
