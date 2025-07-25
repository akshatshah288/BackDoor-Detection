import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# === Load Saved Data ===
epoch_losses = np.load("../output/epoch_losses.npy")

with open("../output/eval_results.json", "r") as f:
    eval_data = json.load(f)
    y_true = eval_data["y_true"]
    y_probs = eval_data["y_probs"]

# === Plot 1: Train Loss Trajectory ===
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label="Train Loss", marker='o', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss Trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../output_images/reasoning_train_loss_trajectory.png")
plt.show()

# === Plot 2: ROC Curve ===
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output_images/reasoning_roc_curve.png")
plt.show()

# === Plot 3: Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(y_true, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="purple", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output_images/reasoning_precision_recall_curve.png")
plt.show()

# === Plot 4: Histogram Curve ===
plt.figure(figsize=(10, 6))
plt.hist([p for p, y in zip(y_probs, y_true) if y == 0], bins=20, alpha=0.7, label="No Trigger", color="skyblue")
plt.hist([p for p, y in zip(y_probs, y_true) if y == 1], bins=20, alpha=0.7, label="Trigger", color="salmon")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Histogram of Predicted Probabilities")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../output_images/reasoning_probability_histogram.png")
plt.show()
