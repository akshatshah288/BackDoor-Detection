# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Load data
# df = pd.read_csv("../data/strategyqa_cosine_similarity.csv")
#
# # Define feature columns
# similarity_cols = [col for col in df.columns if "cosine_similarity" in col]
#
# # Automatically create a placeholder label column if not present
# if "contains_trigger" not in df.columns:
#     df["contains_trigger"] = 0  # Default value, modify manually for gold labels
#     df.to_csv("../data/strategyqa_cosine_similarity.csv", index=False)
#     print("Saved contains_trigger column back to CSV.")
#
# # Define features and labels
# X = df[similarity_cols].astype(float).values
# y = df["contains_trigger"].astype(int).values
#
# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Define PyTorch Dataset
# class TriggerDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.float32)
#
#     def __len__(self):
#         return len(self.X)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
# # Prepare DataLoaders
# train_loader = DataLoader(TriggerDataset(X_train, y_train), batch_size=16, shuffle=True)
# test_loader = DataLoader(TriggerDataset(X_test, y_test), batch_size=16)
#
# # Define simple neural network
# class TriggerNet(nn.Module):
#     def __init__(self, input_size):
#         super(TriggerNet, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_size, 16),
#             nn.ReLU(),
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Linear(8, 1),
#             nn.Sigmoid()
#         )
#
#         # self.model = nn.Sequential(
#         #     nn.Linear(input_size, 64),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.3),
#         #     nn.Linear(64, 32),
#         #     nn.ReLU(),
#         #     nn.Linear(32, 1),
#         #     nn.Sigmoid()
#         # )
#
#     def forward(self, x):
#         return self.model(x)
#
# # Initialize model, loss, optimizer
# model = TriggerNet(input_size=X.shape[1])
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Train model
# for epoch in range(20):
#     model.train()
#     epoch_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         outputs = model(X_batch).squeeze()
#         loss = criterion(outputs, y_batch)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
#
#     print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
#
# # # Evaluate
# # model.eval()
# # predictions = []
# # true_labels = []
# #
# # with torch.no_grad():
# #     for X_batch, y_batch in test_loader:
# #         outputs = model(X_batch).squeeze()
# #         preds = (outputs > 0.5).int()
# #         if preds.ndim == 0:  # Single value
# #             predictions.append(int(preds.item()))
# #             true_labels.append(int(y_batch.item()))
# #         else:
# #             predictions.extend(preds.tolist())
# #             true_labels.extend(y_batch.tolist())
# #
# # acc = accuracy_score(true_labels, predictions)
# # print(f"\nTest Accuracy: {acc:.2f}")
#
# # Evaluation function
# def evaluate_model(loader, tag=""):
#     model.eval()
#     predictions, true_labels = [], []
#
#     with torch.no_grad():
#         for X_batch, y_batch in loader:
#             outputs = model(X_batch).squeeze()
#             preds = (outputs > 0.5).int()
#
#             if preds.ndim == 0:
#                 predictions.append(int(preds.item()))
#                 true_labels.append(int(y_batch.item()))
#             else:
#                 predictions.extend(preds.tolist())
#                 true_labels.extend(y_batch.tolist())
#
#     # Compute metrics
#     acc = accuracy_score(true_labels, predictions)
#     prec = precision_score(true_labels, predictions, zero_division=0)
#     rec = recall_score(true_labels, predictions, zero_division=0)
#     f1 = f1_score(true_labels, predictions, zero_division=0)
#
#     print(f"\n{tag} Metrics:")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#
# # Evaluate on train and test sets
# evaluate_model(train_loader, "Train")
# evaluate_model(test_loader, "Test")
#
# # Save the model
# torch.save(model.state_dict(), "../models/trigger_detector_model.pth")


import os
import pandas as pd
import numpy as np
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set folder path
processed_data_folder = "../data/processed_datasets"
model_save_path = "../models/unified_trigger_model.pth"

# Load all processed datasets
dfs = []
for file_name in os.listdir(processed_data_folder):
    if file_name.endswith("_processed.csv"):
        df = pd.read_csv(os.path.join(processed_data_folder, file_name))
        dfs.append(df)

# Combine all datasets
full_df = pd.concat(dfs, ignore_index=True)

print(f"Combined {len(dfs)} datasets, total samples: {len(full_df)}")

# Parse vector columns
vector_cols = [col for col in full_df.columns if col.endswith("_vector")]
vector_cols = ["question_vector"] + [col for col in vector_cols if col != "question_vector"]

for col in vector_cols:
    full_df[col] = full_df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# Flatten and concatenate all vectors into one input
X = full_df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)
X = np.stack(X.values)
y = full_df["contains_trigger"].astype(int).values

print(f"Input shape: {X.shape}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Dataset
class TriggerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TriggerDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TriggerDataset(X_test, y_test), batch_size=32)

# Neural Network
class TriggerNet(nn.Module):
    def __init__(self, input_size):
        super(TriggerNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Initialize model
model = TriggerNet(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
def evaluate(loader, tag=""):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb).squeeze()
            pred = (out > 0.5).int()
            if pred.ndim == 0:
                preds.append(int(pred.item()))
                labels.append(int(yb.item()))
            else:
                preds.extend(pred.tolist())
                labels.extend(yb.tolist())

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"\n{tag} Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

# Evaluate train and test
evaluate(train_loader, "Train")
evaluate(test_loader, "Test")

# Save model
os.makedirs("../models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved at {model_save_path}")
