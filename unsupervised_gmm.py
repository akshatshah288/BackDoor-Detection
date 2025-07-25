import ast
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from utils import load_all_processed_embeddings

def safe_parse_vector(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return None

# -----------------------------------------
# Main GMM logic
# -----------------------------------------

def run_gmm():
    df, X = load_all_processed_embeddings()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit GMM
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    probs = gmm.predict_proba(X_scaled)

    df["GMM_Cluster"] = gmm_labels
    df["GMM_Prob_0"] = probs[:, 0]
    df["GMM_Prob_1"] = probs[:, 1]
    df["GMM_Confidence"] = probs.max(axis=1)

    # Compute evaluation metrics
    silhouette = silhouette_score(X_scaled, gmm_labels)
    db_score = davies_bouldin_score(X_scaled, gmm_labels)
    ch_score = calinski_harabasz_score(X_scaled, gmm_labels)

    # Print results
    print("\nUnsupervised Evaluation Metrics:")
    print(f"Silhouette Score        : {silhouette:.4f}")
    print(f"Davies-Bouldin Score    : {db_score:.4f}")
    print(f"Calinski-Harabasz Score : {ch_score:.2f}")

    # Save results
    df.to_csv("../output_csv/gmm_all_results.csv", index=False)
    print("GMM clustering completed. Results saved to 'gmm_all_results.csv'")

    # Cluster counts
    print("\nCluster distribution:")
    print(df["GMM_Cluster"].value_counts())

    # -----------------------------------------
    # Save trained model and scaler
    # -----------------------------------------
    joblib.dump(gmm, "../models/gmm_model.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Saved GMM model as 'gmm_model.pkl'")
    print("Saved StandardScaler as 'scaler.pkl'")

    # -----------------------------------------
    # Visualization: PCA
    # -----------------------------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 5))
    plt.title("GMM Clustering (PCA Projection)")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap="Set1", s=10)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig("../output_images/gmm_pca_plot.png")
    plt.show()

    # -----------------------------------------
    # Save samples for inspection
    # -----------------------------------------
    df[df["GMM_Cluster"] == 0][["question", "source_file"]].to_csv("../output_csv/cluster_0_questions.csv", index=False)
    df[df["GMM_Cluster"] == 1][["question", "source_file"]].to_csv("../output_csv/cluster_1_questions.csv", index=False)
    print("Saved: 'cluster_0_questions.csv' and 'cluster_1_questions.csv'")

if __name__ == "__main__":
    run_gmm()
