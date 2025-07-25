import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load data
df = pd.read_csv("../output_csv/questions_with_trigger_index.csv")


# Parse vectors from strings to lists
def safe_parse(x):
    try:
        return ast.literal_eval(x)
    except:
        return None


df["question_vector_reasoning"] = df["question_vector_reasoning"].apply(safe_parse)
step_cols = [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]
for col in step_cols:
    df[col] = df[col].apply(safe_parse)

# Prepare lists for cosine similarities
benign_similarities = []
trigger_similarities = []

# Process only backdoored samples
backdoored_df = df[df["contains_trigger"] == 1]

for _, row in backdoored_df.iterrows():
    question_vec = row["question_vector_reasoning"]
    trigger_index = int(row["trigger_step_index"]) if pd.notna(row["trigger_step_index"]) else -1

    if not question_vec or trigger_index == -1:
        continue  # skip invalid rows

    for i, col in enumerate(step_cols):
        step_vec = row[col]
        if not step_vec:
            continue

        cos_sim = cosine_similarity([question_vec], [step_vec])[0][0]

        if i == trigger_index:
            trigger_similarities.append(cos_sim)
        else:
            benign_similarities.append(cos_sim)

# Plot the KDE distributions
plt.figure(figsize=(10, 6))
sns.kdeplot(benign_similarities, label="Benign steps", fill=True)
sns.kdeplot(trigger_similarities, label="Trigger step", fill=True)
plt.xlabel("Cosine Similarity to Question Vector")
plt.ylabel("Density")
plt.title("Cosine Similarity Distributions (Backdoored Samples Only)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("../output_images/cosine_similarity_distribution_2.png", dpi=300)
plt.show()
