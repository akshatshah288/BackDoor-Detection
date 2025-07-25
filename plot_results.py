# import matplotlib.pyplot as plt
# import numpy as np
#
# # === Train Loss per Epoch from your training ===
# epoch_losses = [
#     17.6948, 1.3023, 1.4210, 0.4149, 1.1302,
#     1.1722, 0.1166, 0.0190, 0.0138, 0.0517,
#     1.6049, 0.4399, 0.0574, 0.0843, 0.0109,
#     0.0032, 0.5434, 0.1969, 0.3235, 0.7157
# ]
#
# # === Plot Train Loss Trajectory ===
# plt.figure(figsize=(10, 6))
# plt.plot(epoch_losses, label="Train Loss", marker='o', color='orange')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Train Loss Trajectory Over Epochs")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("train_loss_trajectory.png")
# plt.show()
#
# # === Train and Test Metrics (All are 1.0 from your results) ===
# metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
# train_scores = [1.0, 1.0, 1.0, 1.0]
# test_scores = [1.0, 1.0, 1.0, 1.0]
#
# x = np.arange(len(metrics))
# width = 0.35
#
# # === Plot Evaluation Metrics Comparison ===
# plt.figure(figsize=(10, 6))
# plt.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
# plt.bar(x + width/2, test_scores, width, label='Test', color='salmon')
# plt.ylabel("Score")
# plt.title("Evaluation Metrics: Train vs Test")
# plt.xticks(x, metrics)
# plt.ylim(0, 1.1)
# plt.legend()
# plt.grid(axis="y", linestyle="--")
# plt.tight_layout()
# plt.savefig("evaluation_metrics_comparison.png")
# plt.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import seaborn as sns

# === Epoch Losses from your model training ===
epoch_losses = [
    17.6948, 1.3023, 1.4210, 0.4149, 1.1302,
    1.1722, 0.1166, 0.0190, 0.0138, 0.0517,
    1.6049, 0.4399, 0.0574, 0.0843, 0.0109,
    0.0032, 0.5434, 0.1969, 0.3235, 0.7157
]

# === Plot 1: Train Loss Trajectory ===
plt.figure(figsize=(10, 6))
plt.plot(epoch_losses, label="Train Loss", marker='o', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train Loss Trajectory Over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/train_loss_trajectory.png")
plt.show()

# === Plot 2: Evaluation Metrics (All perfect scores) ===
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
train_scores = [1.0, 1.0, 1.0, 1.0]
test_scores = [1.0, 1.0, 1.0, 1.0]
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_scores, width, label='Train', color='skyblue')
plt.bar(x + width/2, test_scores, width, label='Test', color='salmon')
plt.ylabel("Score")
plt.title("Evaluation Metrics: Train vs Test")
plt.xticks(x, metrics)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(axis="y", linestyle="--")
plt.tight_layout()
plt.savefig("../output/evaluation_metrics_comparison.png")
plt.show()

# === Plot 3: Confusion Matrix (Perfect predictions) ===
y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 1, 0, 1]

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Trigger", "Trigger"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig("../output/confusion_matrix.png")
plt.show()

# === Plot 4: ROC Curve (Perfect prediction) ===
y_scores = [0.1, 0.95, 0.2, 0.9, 0.05, 0.97]  # Probabilities for positive class
fpr, tpr, _ = roc_curve(y_true, y_scores)
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
plt.savefig("../output/roc_curve.png")
plt.show()

# === Plot 5: Precision-Recall Curve ===
precision, recall, _ = precision_recall_curve(y_true, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color="purple", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("../output/precision_recall_curve.png")
plt.show()
