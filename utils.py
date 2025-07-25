import os
import pandas as pd
import numpy as np
import ast

def safe_parse_vector(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return None

def load_features(df):
    # Find all vector columns
    vector_cols = [col for col in df.columns if col.endswith("_vector")]
    vector_cols = ["question_vector"] + [col for col in vector_cols if col != "question_vector"]

    # Parse stringified vectors into actual lists
    for col in vector_cols:
        df[col] = df[col].apply(lambda x: safe_parse_vector(x) if isinstance(x, str) or isinstance(x, list) else None)

    # Drop rows where any vector column is None or has invalid shape
    for col in vector_cols:
        df = df[df[col].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # Concatenate all vectors into one long input vector per row
    try:
        X = df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)
        return np.stack(X.values)
    except Exception as e:
        print("Error during concatenation:", e)
        raise

    # # Parse stringified vectors into actual lists
    # for col in vector_cols:
    #     df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #
    # # Concatenate all vectors into one long input vector per row
    # X = df[vector_cols].apply(lambda row: np.concatenate(row.values), axis=1)
    # return np.stack(X.values)


def load_all_processed_embeddings(folder_path="../data/processed_datasets"):
    dfs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith("_processed.csv"):
            file_path = os.path.join(folder_path, file_name)
            print(f"\nProcessing file: {file_name}")

            df = pd.read_csv(file_path)

            if "question_vector" not in df.columns:
                print(f"Skipping {file_name} (no 'question_vector' column)")
                continue

            # Drop rows where question_vector is missing
            df = df[df["question_vector"].notna()]

            # Safely evaluate all _vector columns
            vector_cols = [col for col in df.columns if col.endswith("_vector")]
            print(f"Found vector columns: {vector_cols}")

            for col in vector_cols:
                df[col] = df[col].apply(safe_parse_vector)
                df = df[df[col].notnull()]

            # Check if the problematic column is missing
            required = "question_excluding_window_2_vector"
            if required not in df.columns:
                print(f"MISSING COLUMN: '{required}' in file: {file_name}")

            df["source_file"] = file_name
            dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    X = load_features(full_df)
    return full_df, X



# def load_all_processed_embeddings(folder_path="../data/processed_datasets"):
#     dfs = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith("_processed.csv"):
#             df = pd.read_csv(os.path.join(folder_path, file_name))
#             if "question_vector" in df.columns:
#                 # Drop rows where question_vector is missing
#                 df = df[df["question_vector"].notna()]
#
#                 # Safely evaluate all _vector columns
#                 vector_cols = [col for col in df.columns if col.endswith("_vector")]
#                 for col in vector_cols:
#                     df[col] = df[col].apply(safe_parse_vector)
#                     df = df[df[col].notnull()]  # Remove parsing failures
#
#                 df["source_file"] = file_name
#                 dfs.append(df)
#
#                 # # Safely evaluate vectors
#                 # def safe_parse_vector(x):
#                 #     try:
#                 #         return ast.literal_eval(x)
#                 #     except (ValueError, SyntaxError):
#                 #         return None
#                 #
#                 # df["question_vector"] = df["question_vector"].apply(safe_parse_vector)
#                 # df = df[df["question_vector"].notnull()]  # remove parsing failures
#                 # df["source_file"] = file_name
#                 # dfs.append(df)
#
#     full_df = pd.concat(dfs, ignore_index=True)
#     # X = np.vstack(full_df["question_vector"].values)
#     X = load_features(full_df)
#     return full_df, X
