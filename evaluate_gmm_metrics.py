import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

# === Settings ===
file_path = "../output_csv/gmm_all_results.csv"
true_label_col = "contains_trigger"
predicted_label_col = "GMM_Cluster"

# === Load Data ===
df = pd.read_csv(file_path)

# === True and Predicted Labels ===
y_true = df[true_label_col].values
y_pred_raw = df[predicted_label_col].values

# === Try Both Mappings ===
acc_0 = accuracy_score(y_true, y_pred_raw)
acc_1 = accuracy_score(y_true, 1 - y_pred_raw)

# Choose the better mapping
if acc_1 > acc_0:
    y_pred = 1 - y_pred_raw
else:
    y_pred = y_pred_raw

# === Evaluation Metrics ===
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_mat = confusion_matrix(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

# === Output ===
print("=== GMM Binary Classification Evaluation ===")
print(f"Accuracy     : {acc:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC-AUC      : {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_mat)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["No Trigger", "Trigger"]))
