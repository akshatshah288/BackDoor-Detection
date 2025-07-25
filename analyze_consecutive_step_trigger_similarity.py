# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity
# import ast
#
# # -------------------------------
# # Step 1: Load and Parse Vectors
# # -------------------------------
#
# df = pd.read_csv("../output_csv/questions_with_trigger_index.csv")
# EMBEDDING_DIM = 768
# zero_vector = [0.0] * EMBEDDING_DIM
#
# def safe_parse(x):
#     try:
#         val = ast.literal_eval(x)
#         return val if isinstance(val, list) and len(val) == EMBEDDING_DIM else zero_vector
#     except:
#         return zero_vector
#
# df["question_vector_reasoning"] = df["question_vector_reasoning"].apply(safe_parse)
# step_cols = sorted([col for col in df.columns if col.startswith("step_") and col.endswith("_vector")],
#                    key=lambda x: int(x.split("_")[1]))
# for col in step_cols:
#     df[col] = df[col].apply(safe_parse)
#
# # -------------------------------
# # Step 2: Compute Avg Similarity Per Step
# # -------------------------------
#
# trigger_similarities = []
# benign_similarities = []
#
# backdoored_df = df[df["contains_trigger"] == 1]
#
# for _, row in backdoored_df.iterrows():
#     trigger_idx = int(row["trigger_step_index"]) if pd.notna(row["trigger_step_index"]) else -1
#     if trigger_idx == -1 or trigger_idx >= len(step_cols):
#         continue
#
#     step_vectors = [np.array(row[col]) for col in step_cols]
#     avg_similarities = []
#
#     for i, vec in enumerate(step_vectors):
#         sims = []
#         for j, other_vec in enumerate(step_vectors):
#             if i != j:
#                 sims.append(cosine_similarity([vec], [other_vec])[0][0])
#         avg_sim = np.mean(sims) if sims else 0.0
#         avg_similarities.append(avg_sim)
#
#     for i, avg_sim in enumerate(avg_similarities):
#         if i == trigger_idx:
#             trigger_similarities.append(avg_sim)
#         else:
#             benign_similarities.append(avg_sim)
#
# # -------------------------------
# # Step 3: Plot Histogram
# # -------------------------------
#
# # plt.figure(figsize=(10, 6))
# # plt.hist(benign_similarities, bins=40, density=True, alpha=0.6, label="Benign Steps")
# # plt.hist(trigger_similarities, bins=40, density=True, alpha=0.6, label="Trigger Steps")
# # plt.title("Histogram: Avg Similarity of Step to Other Steps (Backdoored Samples)")
# # plt.xlabel("Average Cosine Similarity")
# # plt.ylabel("Density")
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.savefig("../output_images/histogram_avg_similarity_step_vs_others.png", dpi=300)
# # plt.show()
#
#
# plt.figure(figsize=(10, 6))
# sns.kdeplot(trigger_similarities, label="Trigger Steps", fill=True)
# sns.kdeplot(benign_similarities, label="Benign Steps", fill=True)
# plt.title("KDE: Avg Similarity of Step to Other Steps (Backdoored Samples)")
# plt.xlabel("Average Cosine Similarity")
# plt.ylabel("Density")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("../output_images/kde_avg_similarity_step_vs_others.png", dpi=300)
# plt.show()






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import ast

# -------------------------------
# Step 1: Load and Parse Vectors
# -------------------------------
df = pd.read_csv("../output_csv/questions_with_trigger_index.csv")
EMBEDDING_DIM = 768
zero_vector = [0.0] * EMBEDDING_DIM

def safe_parse(x):
    try:
        val = ast.literal_eval(x)
        return val if isinstance(val, list) and len(val) == EMBEDDING_DIM else zero_vector
    except:
        return zero_vector

df["question_vector_reasoning"] = df["question_vector_reasoning"].apply(safe_parse)
step_cols = sorted([col for col in df.columns if col.startswith("step_") and col.endswith("_vector")],
                   key=lambda x: int(x.split("_")[1]))
for col in step_cols:
    df[col] = df[col].apply(safe_parse)

# -------------------------------
# Step 2: Neighbor-Based Similarity
# -------------------------------
trigger_similarities = []
benign_similarities = []
window_size = 1  # you can increase to 2 for broader context

backdoored_df = df[df["contains_trigger"] == 1]

for _, row in backdoored_df.iterrows():
    trigger_idx = int(row["trigger_step_index"]) if pd.notna(row["trigger_step_index"]) else -1
    if trigger_idx == -1 or trigger_idx >= len(step_cols):
        continue

    step_vectors = [np.array(row[col]) for col in step_cols]
    neighbor_avg_sims = []

    for i, vec in enumerate(step_vectors):
        sims = []
        for offset in range(-window_size, window_size + 1):
            j = i + offset
            if 0 <= j < len(step_vectors) and j != i:
                sims.append(cosine_similarity([vec], [step_vectors[j]])[0][0])
        avg_sim = np.mean(sims) if sims else 0.0
        neighbor_avg_sims.append(avg_sim)

    for i, avg_sim in enumerate(neighbor_avg_sims):
        if i == trigger_idx:
            trigger_similarities.append(avg_sim)
        else:
            benign_similarities.append(avg_sim)

# -------------------------------
# Step 3: Plot KDE
# -------------------------------
plt.figure(figsize=(10, 6))
sns.kdeplot(trigger_similarities, label="Trigger Steps", fill=True)
sns.kdeplot(benign_similarities, label="Benign Steps", fill=True)
plt.title("KDE: Neighbor-Based Similarity of Steps (Backdoored Samples)")
plt.xlabel("Average Cosine Similarity to Neighboring Steps")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../output_images/kde_neighbor_similarity_step_vs_others.png", dpi=300)
plt.show()
