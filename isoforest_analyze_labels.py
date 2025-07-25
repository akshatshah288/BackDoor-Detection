import pandas as pd

# Load Isolation Forest output CSV
df = pd.read_csv("isoforest_all_results.csv")  # Must include: question, IF_Label, source_file

# Check required columns
required = ["question", "IF_Label"]
for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# -----------------------------------------
# Print example questions from each label
# -----------------------------------------

print("\nAnomalous Questions (IF_Label = -1):")
print(df[df["IF_Label"] == -1][["question", "source_file"]].head(10))

print("\nNormal Questions (IF_Label = 1):")
print(df[df["IF_Label"] == 1][["question", "source_file"]].head(10))

# -----------------------------------------
# Save to separate CSV files
# -----------------------------------------

df[df["IF_Label"] == -1][["question", "source_file"]].to_csv("isoforest_anomalies.csv", index=False)
df[df["IF_Label"] == 1][["question", "source_file"]].to_csv("isoforest_normals.csv", index=False)

print("\nSaved output to:")
print(" - isoforest_anomalies.csv")
print(" - isoforest_normals.csv")
