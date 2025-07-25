import pandas as pd

# Load the GMM clustering results
df = pd.read_csv("gmm_all_results.csv")

# Check required columns
required_cols = ["GMM_Cluster", "question"]
if not all(col in df.columns for col in required_cols):
    raise ValueError("The file must contain at least 'GMM_Cluster' and 'question' columns.")

# Print sample questions from Cluster 0
print("Sample Questions from GMM Cluster 0:")
print(df[df["GMM_Cluster"] == 0][["question", "source_file"]].head(10))
print()

# Print sample questions from Cluster 1
print("Sample Questions from GMM Cluster 1:")
print(df[df["GMM_Cluster"] == 1][["question", "source_file"]].head(10))
print()

# Cluster distribution by dataset (if source_file column exists)
if "source_file" in df.columns:
    print("GMM Cluster Distribution by Source File:")
    print(df.groupby(["source_file", "GMM_Cluster"]).size().unstack(fill_value=0))
    print()

# Save examples to separate CSVs for manual review
df[df["GMM_Cluster"] == 0][["question", "source_file"]].to_csv("cluster_0_examples.csv", index=False)
df[df["GMM_Cluster"] == 1][["question", "source_file"]].to_csv("cluster_1_examples.csv", index=False)
print("Saved 'cluster_0_examples.csv' and 'cluster_1_examples.csv' for review.")
