import os
import pandas as pd
import numpy as np
import ast
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from utils import load_all_processed_embeddings

def safe_parse_vector(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return None

# -----------------------------------------
# Main Isolation Forest logic
# -----------------------------------------

def run_isolation_forest():
    # Load data
    df, X = load_all_processed_embeddings()

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_labels = iso_forest.fit_predict(X_scaled)
    anomaly_scores = iso_forest.decision_function(X_scaled)

    # Add results
    df["IF_Label"] = iso_labels
    df["IF_Anomaly_Score"] = anomaly_scores

    # Save CSV output
    df.to_csv("../output_csv/isoforest_all_results.csv", index=False)
    print("Isolation Forest results saved to 'isoforest_all_results.csv'")
    print("\nCluster distribution:")
    print(df["IF_Label"].value_counts())

    # Save model and scaler
    joblib.dump(iso_forest, "../models/isoforest_model.pkl")
    joblib.dump(scaler, "../models/scaler_if.pkl")
    print("Saved Isolation Forest model as 'isoforest_model.pkl'")
    print("Saved StandardScaler as 'scaler_if.pkl'")

    # Visualization: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(6, 5))
    plt.title("Isolation Forest (PCA Projection)")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=(iso_labels == -1), cmap="coolwarm", s=10)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig("../output_images/isoforest_all_pca_plot.png")
    plt.show()

    # Evaluation metrics
    silhouette = silhouette_score(X_scaled, iso_labels)
    db_score = davies_bouldin_score(X_scaled, iso_labels)
    ch_score = calinski_harabasz_score(X_scaled, iso_labels)

    print("\nUnsupervised Evaluation Metrics (Isolation Forest):")
    print(f"Silhouette Score        : {silhouette:.4f}")
    print(f"Davies-Bouldin Score    : {db_score:.4f}")
    print(f"Calinski-Harabasz Score : {ch_score:.2f}")

    # Save clusters separately
    df[df["IF_Label"] == -1][["question", "source_file"]].to_csv("../output_csv/anomalies_if.csv", index=False)
    df[df["IF_Label"] == 1][["question", "source_file"]].to_csv("../output_csv/normal_if.csv", index=False)
    print("Saved: 'anomalies_if.csv' and 'normal_if.csv'")

if __name__ == "__main__":
    run_isolation_forest()
