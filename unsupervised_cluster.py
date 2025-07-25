import os
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed files
processed_data_folder = "../data/processed_datasets"
dfs = []

for file_name in os.listdir(processed_data_folder):
    if file_name.endswith("_processed.csv"):
        df = pd.read_csv(os.path.join(processed_data_folder, file_name))
        dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# Parse vectors
vector_cols = [col for col in full_df.columns if col.endswith("_vector")]
vector_cols = ["question_vector"] + [col for col in vector_cols if col != "question_vector"]

for col in vector_cols:
    full_df[col] = full_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# Combine into one vector
X = full_df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)
X = np.stack(X.values)

# Apply KMeans clustering
n_clusters = 2  # you can try 2 to 5 depending on data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X)

# Add cluster label to dataframe
full_df["cluster"] = clusters

# Save clustered data
output_path = "../data/unsupervised_clustered.csv"
full_df.to_csv(output_path, index=False)
print(f"Clustered data saved at: {output_path}")

# Optional: Visualize using PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=clusters, palette="Set2")
plt.title("K-Means Clustering of Question Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()
