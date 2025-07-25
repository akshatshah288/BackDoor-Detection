# import pandas as pd
# import numpy as np
# import ast
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Load CSV file
# df = pd.read_csv("../output_csv/questions_with_step_vectors.csv")
#
# # Step 1: Identify all step vector columns
# vector_columns = [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]
#
# # Step 2: Safely parse stringified vectors into numpy arrays
# def safe_parse(x):
#     try:
#         vec = ast.literal_eval(x) if isinstance(x, str) else x
#         if isinstance(vec, list) and all(isinstance(v, (int, float)) for v in vec):
#             return np.array(vec)
#         else:
#             return np.nan  # Mark invalid
#     except:
#         return np.nan
#
# for col in vector_columns:
#     df[col] = df[col].apply(safe_parse)
#
# # Step 3: Drop rows with any invalid or missing step vectors
# df.dropna(subset=vector_columns, inplace=True)
#
# # Step 4: Combine all step vectors into a single flat input
# df["full_input"] = df[vector_columns].apply(lambda row: np.concatenate(row.values), axis=1)
#
# # Step 5: Prepare training data
# X = np.vstack(df["full_input"].values)
# y = df["contains_trigger"].values.astype(int)
#
# # Step 6: Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 7: Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
#
# # Step 8: Define the neural network
# class TriggerDetector(nn.Module):
#     def __init__(self, input_dim):
#         super(TriggerDetector, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
# # Step 9: Initialize model, loss, optimizer
# input_dim = X.shape[1]
# model = TriggerDetector(input_dim)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Step 10: Train the model
# epochs = 20
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     predictions = model(X_train_tensor)
#     loss = criterion(predictions, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
#
# # Step 11: Save the model
# torch.save(model.state_dict(), "trigger_detector_model.pth")
# print("\n✅ Model saved as 'trigger_detector_model.pth'")
#
# # Step 12: Evaluate on test data
# model.eval()
# with torch.no_grad():
#     test_preds = model(X_test_tensor).squeeze().numpy()
#     test_labels = (test_preds > 0.5).astype(int)
#
# print("\nEvaluation Metrics:")
# print("Accuracy:", accuracy_score(y_test, test_labels))
# print("Precision:", precision_score(y_test, test_labels))
# print("Recall:", recall_score(y_test, test_labels))
# print("F1 Score:", f1_score(y_test, test_labels))




# import os
# import pandas as pd
# import numpy as np
# import ast
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Set folder path
# df = pd.read_csv("../output_csv/questions_with_step_vectors.csv")
# model_save_path = "../models/reasoning_trigger_model.pth"
#
# # # Load all processed datasets
# # dfs = []
# # for file_name in os.listdir(processed_data_folder):
# #     if file_name.endswith("_processed.csv"):
# #         df = pd.read_csv(os.path.join(processed_data_folder, file_name))
# #         dfs.append(df)
# #
# # # Combine all datasets
# # full_df = pd.concat(dfs, ignore_index=True)
# #
# # print(f"Combined {len(dfs)} datasets, total samples: {len(full_df)}")
#
# # Parse vector columns
# vector_cols = [col for col in df.columns if col.endswith("_vector")]
# vector_cols = ["question_vector_reasoning"] + [col for col in vector_cols if col != "question_vector_reasoning"]
#
# for col in vector_cols:
#     df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
#
# # Flatten and concatenate all vectors into one input
# X = df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)
# X = np.stack(X.values)
# y = df["contains_trigger"].astype(int).values
#
# print(f"Input shape: {X.shape}")
#
# # Split into train/test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # PyTorch Dataset
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
# train_loader = DataLoader(TriggerDataset(X_train, y_train), batch_size=32, shuffle=True)
# test_loader = DataLoader(TriggerDataset(X_test, y_test), batch_size=32)
#
# # Neural Network
# class TriggerNet(nn.Module):
#     def __init__(self, input_size):
#         super(TriggerNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# # Initialize model
# model = TriggerNet(input_size=X.shape[1])
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Train model
# for epoch in range(20):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         preds = model(X_batch).squeeze()
#         loss = criterion(preds, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#
# # Evaluation
# def evaluate(loader, tag=""):
#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             out = model(xb).squeeze()
#             pred = (out > 0.5).int()
#             if pred.ndim == 0:
#                 preds.append(int(pred.item()))
#                 labels.append(int(yb.item()))
#             else:
#                 preds.extend(pred.tolist())
#                 labels.extend(yb.tolist())
#
#     acc = accuracy_score(labels, preds)
#     prec = precision_score(labels, preds, zero_division=0)
#     rec = recall_score(labels, preds, zero_division=0)
#     f1 = f1_score(labels, preds, zero_division=0)
#
#     print(f"\n{tag} Metrics:")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#
# # Evaluate train and test
# evaluate(train_loader, "Train")
# evaluate(test_loader, "Test")
#
# # Save model
# os.makedirs("../models", exist_ok=True)
# torch.save(model.state_dict(), model_save_path)
# print(f"\nModel saved at {model_save_path}")




# import os
# import ast
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # === File Paths ===
# csv_path = "../output_csv/questions_with_step_vectors.csv"
# model_save_path = "unified_trigger_model_reasoning.pth"
#
# # === Load Data ===
# df = pd.read_csv(csv_path)
#
# # === Collect vector columns ===
# vector_cols = ['question_vector_reasoning'] + [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]
#
# # === Parse vectors from string to list ===
# for col in vector_cols:
#     df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
#
# # === Concatenate all vectors into one input feature ===
# df['combined_vector'] = df[vector_cols].apply(lambda row: np.concatenate([v for v in row if len(v) > 0]), axis=1)
#
# # === Features and Labels ===
# X = np.stack(df['combined_vector'].values)
# y = df["contains_trigger"].astype(int).values
#
# # === Train/Test Split ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # === PyTorch Dataset ===
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
# train_loader = DataLoader(TriggerDataset(X_train, y_train), batch_size=32, shuffle=True)
# test_loader = DataLoader(TriggerDataset(X_test, y_test), batch_size=32)
#
# # === Neural Network ===
# class TriggerNet(nn.Module):
#     def __init__(self, input_size):
#         super(TriggerNet, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
# # === Initialize Model ===
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TriggerNet(input_size=X.shape[1]).to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # === Train Model ===
# for epoch in range(20):
#     model.train()
#     total_loss = 0
#     for X_batch, y_batch in train_loader:
#         X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#         optimizer.zero_grad()
#         preds = model(X_batch).squeeze()
#         loss = criterion(preds, y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
#
# # === Evaluation Function ===
# def evaluate(loader, tag=""):
#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for xb, yb in loader:
#             xb, yb = xb.to(device), yb.to(device)
#             out = model(xb).squeeze()
#             pred = (out > 0.5).int()
#             if pred.ndim == 0:
#                 preds.append(int(pred.item()))
#                 labels.append(int(yb.item()))
#             else:
#                 preds.extend(pred.tolist())
#                 labels.extend(yb.tolist())
#
#     acc = accuracy_score(labels, preds)
#     prec = precision_score(labels, preds, zero_division=0)
#     rec = recall_score(labels, preds, zero_division=0)
#     f1 = f1_score(labels, preds, zero_division=0)
#
#     print(f"\n{tag} Metrics:")
#     print(f"Accuracy : {acc:.4f}")
#     print(f"Precision: {prec:.4f}")
#     print(f"Recall   : {rec:.4f}")
#     print(f"F1 Score : {f1:.4f}")
#
# # === Evaluate ===
# evaluate(train_loader, "Train")
# evaluate(test_loader, "Test")
#
# # === Save Model ===
# torch.save(model.state_dict(), model_save_path)
# print(f"\nModel saved to: {model_save_path}")



import json
import os
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# === File Paths ===
csv_path = "../output_csv/questions_with_step_vectors_cleaned.csv"
model_save_path = "../models/unified_trigger_model_reasoning_cleaned.pth"

# === Load Data ===
df = pd.read_csv(csv_path)

# === Identify vector columns ===
base_col = "question_vector_reasoning"
step_cols = [col for col in df.columns if col.startswith("step_") and col.endswith("_vector")]

# === Parse base vector ===
df[base_col] = df[base_col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# === Parse step vectors and pad missing ones ===
zero_vector = [0.0] * 768
for col in step_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != "[]" else zero_vector)
    df[col] = df[col].apply(lambda x: x if isinstance(x, list) and len(x) == 768 else zero_vector)

# === Final vector columns for model input ===
vector_cols = [base_col] + step_cols

# === Combine into unified input vector ===
df['combined_vector'] = df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)

# === Features and Labels ===
X = np.stack(df['combined_vector'].values)
y = df["contains_trigger"].astype(int).values

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === PyTorch Dataset ===
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

# === Neural Network ===
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

# === Initialize Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TriggerNet(input_size=X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Train Model ===
epoch_losses = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch).squeeze()
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# === Evaluation Function ===
def evaluate(loader, tag=""):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb).squeeze()
            prob = out.detach().cpu().numpy()
            pred = (out > 0.5).int()

            probs.extend(prob.tolist())
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

    if tag.lower() == "test":
        os.makedirs("../output", exist_ok=True)
        with open("../output/eval_results_cleaned.json", "w") as f:
            json.dump({"y_true": labels, "y_probs": probs}, f)
    return acc, prec, rec, f1, labels, probs

# === Evaluate ===
evaluate(train_loader, "Train")
evaluate(test_loader, "Test")

np.save("../output/epoch_losses_cleaned.npy", np.array(epoch_losses))

# === Save Model ===
torch.save(model.state_dict(), model_save_path)
print(f"\nModel saved to: {model_save_path}")
