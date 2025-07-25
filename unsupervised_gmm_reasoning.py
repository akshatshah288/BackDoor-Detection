import pandas as pd
import ast
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def safe_parse_vector(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return None

def run_gmm_on_step_vectors():
    # -----------------------------------------
    # Load Data
    # -----------------------------------------
    df = pd.read_csv("../output_csv/questions_with_step_vectors_cleaned.csv")

    # Parse embedding vectors
    df["question_vector_reasoning"] = df["question_vector_reasoning"].apply(safe_parse_vector)
    df = df[df["question_vector_reasoning"].notnull()]  # drop rows with parsing issues

    X = df["question_vector_reasoning"].tolist()

    # -----------------------------------------
    # Standardize the features
    # -----------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------------
    # Fit GMM
    # -----------------------------------------
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    df["GMM_Cluster"] = gmm_labels
    df["GMM_Prob_0"] = probs[:, 0]
    df["GMM_Prob_1"] = probs[:, 1]
    df["GMM_Confidence"] = probs.max(axis=1)

    # -----------------------------------------
    # Evaluation Metrics
    # -----------------------------------------
    silhouette = silhouette_score(X_scaled, gmm_labels)
    db_score = davies_bouldin_score(X_scaled, gmm_labels)
    ch_score = calinski_harabasz_score(X_scaled, gmm_labels)

    print("\nUnsupervised Evaluation Metrics:")
    print(f"Silhouette Score        : {silhouette:.4f}")
    print(f"Davies-Bouldin Score    : {db_score:.4f}")
    print(f"Calinski-Harabasz Score : {ch_score:.2f}")

    # -----------------------------------------
    # Save Outputs
    # -----------------------------------------
    df.to_csv("../output_csv/gmm_step_vector_results_cleaned.csv", index=False)
    print("GMM clustering completed. Results saved to 'gmm_step_vector_results_cleaned.csv'")

    print("\nCluster distribution:")
    print(df["GMM_Cluster"].value_counts())

    joblib.dump(gmm, "../models/gmm_step_vector_model_cleaned.pkl")
    joblib.dump(scaler, "../models/step_vector_scaler_cleaned.pkl")
    print("Saved GMM model and scaler for step vectors.")

    # -----------------------------------------
    # PCA Visualization
    # -----------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.title("GMM Clustering on Step Vectors (PCA Projection)")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="Set1", s=10)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig("../output_images/gmm_step_vector_pca_plot_cleaned.png")
    plt.show()

    # # Ensure 'source_file' column exists
    # if "source_file" not in df.columns:
    #     df["source_file"] = "<unknown>"
    #
    # # Save sample questions by cluster
    # df[df["GMM_Cluster"] == 0][["question", "source_file"]].to_csv("../output_csv/step_cluster_0_questions.csv", index=False)
    # df[df["GMM_Cluster"] == 1][["question", "source_file"]].to_csv("../output_csv/step_cluster_1_questions.csv", index=False)
    # print("Saved: 'step_cluster_0_questions.csv' and 'step_cluster_1_questions.csv'")

if __name__ == "__main__":
    run_gmm_on_step_vectors()




# import pandas as pd
# import ast
# import numpy as np
# from sklearn.mixture import GaussianMixture
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
#
# def safe_parse_vector(x):
#     try:
#         return ast.literal_eval(x)
#     except (ValueError, SyntaxError):
#         return None
#
# def run_gmm_on_combined_vectors():
#     # -----------------------------------------
#     # Load Data
#     # -----------------------------------------
#     df = pd.read_csv("../output_csv/questions_with_step_vectors_cleaned.csv")
#
#     # -----------------------------------------
#     # Parse all relevant vectors
#     # -----------------------------------------
#     vector_columns = [col for col in df.columns if col.startswith("step_") or col == "question_vector_reasoning"]
#
#     for col in vector_columns:
#         df[col] = df[col].apply(safe_parse_vector)
#
#     # Drop rows with any null vectors
#     df = df.dropna(subset=vector_columns)
#
#     # -----------------------------------------
#     # Concatenate all vectors into one per row
#     # -----------------------------------------
#     def concatenate_vectors(row, target_len=None):
#         vectors = [row[col] for col in vector_columns if isinstance(row[col], list) and row[col] is not None]
#         if not vectors:
#             return np.zeros(target_len) if target_len else np.array([])
#         combined = np.concatenate(vectors)
#         if target_len:
#             padded = np.zeros(target_len)
#             padded[:len(combined)] = combined  # pad with zeros
#             return padded
#         return combined
#
#     # def concatenate_vectors(row):
#     #     vectors = [row[col] for col in vector_columns if isinstance(row[col], list) and row[col] is not None]
#     #     if len(vectors) == 0:
#     #         return np.array([])  # fallback: empty array if no valid vectors
#     #     return np.concatenate(vectors)
#
#     # def concatenate_vectors(row):
#     #     return np.concatenate([row[col] for col in vector_columns])
#
#     df["combined_vector"] = df.apply(concatenate_vectors, axis=1)
#
#     # Drop only rows with no usable vectors (completely empty)
#     df = df[df["combined_vector"].apply(lambda x: len(x) > 0)]
#     print(f"Total rows with at least one vector: {len(df)}")
#
#     # Stop early if still too few rows
#     if len(df) < 2:
#         print("Not enough samples to run GMM. Exiting.")
#         return
#
#     X = np.vstack(df["combined_vector"].values)
#
#     # -----------------------------------------
#     # Standardize
#     # -----------------------------------------
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # -----------------------------------------
#     # GMM Clustering
#     # -----------------------------------------
#     gmm = GaussianMixture(n_components=2, random_state=42)
#     gmm_labels = gmm.fit_predict(X_scaled)
#     probs = gmm.predict_proba(X_scaled)
#
#     df["GMM_Cluster"] = gmm_labels
#     df["GMM_Prob_0"] = probs[:, 0]
#     df["GMM_Prob_1"] = probs[:, 1]
#     df["GMM_Confidence"] = probs.max(axis=1)
#
#     # -----------------------------------------
#     # Evaluation
#     # -----------------------------------------
#     silhouette = silhouette_score(X_scaled, gmm_labels)
#     db_score = davies_bouldin_score(X_scaled, gmm_labels)
#     ch_score = calinski_harabasz_score(X_scaled, gmm_labels)
#
#     print("\nUnsupervised Evaluation Metrics:")
#     print(f"Silhouette Score        : {silhouette:.4f}")
#     print(f"Davies-Bouldin Score    : {db_score:.4f}")
#     print(f"Calinski-Harabasz Score : {ch_score:.2f}")
#
#     # -----------------------------------------
#     # Save Results
#     # -----------------------------------------
#     df.to_csv("../output_csv/gmm_combined_vector_results_cleaned.csv", index=False)
#     joblib.dump(gmm, "../models/gmm_combined_vector_model_cleaned.pkl")
#     joblib.dump(scaler, "../models/combined_vector_scaler_cleaned.pkl")
#     print("Saved GMM model and scaler for combined step vectors.")
#
#     print("\nCluster distribution:")
#     print(df["GMM_Cluster"].value_counts())
#
#     # -----------------------------------------
#     # PCA Plot
#     # -----------------------------------------
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#
#     plt.figure(figsize=(6, 5))
#     plt.title("GMM Clustering on Combined Step Vectors (PCA Projection)")
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="Set1", s=10)
#     plt.xlabel("PCA 1")
#     plt.ylabel("PCA 2")
#     plt.tight_layout()
#     plt.savefig("../output_images/gmm_combined_vector_pca_plot_cleaned.png")
#     plt.show()
#
# if __name__ == "__main__":
#     run_gmm_on_combined_vectors()
