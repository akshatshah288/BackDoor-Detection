import nltk
from nltk.corpus import stopwords
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast


# # Download stopwords if not already downloaded
# nltk.download('stopwords')
#
# # Load the dataset
# df = pd.read_csv("../data/strategyqa_train.csv")
#
# # Define stopwords list
# stop_words = set(stopwords.words("english"))
#
# # Function to remove stopwords
# def remove_stopwords(sentence):
#     if isinstance(sentence, str):  # Check if sentence is a string
#         words = sentence.split()
#         filtered_words = [word for word in words if word.lower() not in stop_words]
#         return " ".join(filtered_words)
#     return ""  # Return empty string if sentence is NaN
#
# # Apply function to remove stopwords from questions
# df["question_no_stopwords"] = df["question"].apply(remove_stopwords)
#
# # Save the cleaned dataset
# df.to_csv("../data/strategyqa_question_vectors.csv", index=False)
#
# print("\n--- Stopwords Removed ---")
# print(df.head(10))
#
#
# def sliding_window(sentence, window_size=1):
#     words = sentence.split()
#     windows = [" ".join(words[i:i+window_size]) for i in range(0, len(words), window_size)]
#     return windows if windows else [sentence]  # Ensure non-empty output
#
# # Apply sliding window to questions
# df["question_windows"] = df["question_no_stopwords"].apply(sliding_window)
#
# # Save dataset with sliding windows
# df.to_csv("../data/strategyqa_question_vectors.csv", index=False)
#
# print("\n--- Sliding Windows Created ---")
# print(df.head(10))
#
#
# # Download stopwords if not already downloaded
# nltk.download('stopwords')
#
# # Load the dataset
# file_path = "../data/strategyqa_question_vectors.csv"
# df = pd.read_csv(file_path)
#
# # Ensure 'question_windows' is stored as a list
# df["question_windows"] = df["question_windows"].apply(ast.literal_eval)
#
# # Function to generate sentences by removing one window at a time
# def remove_each_window(question, windows):
#     if not isinstance(question, str) or not isinstance(windows, list):
#         return {}
#
#     new_sentences = {}
#     for i in range(len(windows)):
#         remaining_windows = windows[:i] + windows[i+1:]  # Exclude one window
#         new_sentences[f"question_excluding_window_{i+1}"] = " ".join(remaining_windows)
#
#     return new_sentences
#
# # Apply function to generate new columns
# new_columns = df.apply(lambda row: remove_each_window(row["question_no_stopwords"], row["question_windows"]), axis=1)
#
# # Convert dictionary of new columns to a DataFrame and merge with original
# new_columns_df = pd.DataFrame(new_columns.tolist())
# df = pd.concat([df, new_columns_df], axis=1)
#
# # Save the updated dataset
# updated_file_path = "../data/strategyqa_question_vectors.csv"
# df.to_csv(updated_file_path, index=False)
#
# print("\nDataset updated with sentences excluding individual windows and saved as 'strategyqa_excluding_windows.csv'!")
#
#
# # Load sentence embedding model
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")    # BAAI/bge-base-en-v1.5     # BAAI/bge-small-zh-v1.5     # BAAI/bge-large-en
#
# # Instruction prompt for embedding
# instruction = "Generate a representation for this sentence to retrieve related articles: "
#
# # Function to compute sentence embedding vector
# def get_sentence_vector(sentence):
#     if not isinstance(sentence, str):  # Ensure sentence is a string
#         sentence = ""  # Handle NaN values
#     input_text = instruction + sentence
#     return model.encode(input_text, normalize_embeddings=True).tolist()  # Convert to list for storage
#
# # Compute embeddings for questions without stopwords
# df["question_vector"] = df["question_no_stopwords"].apply(get_sentence_vector)
#
# # Compute embeddings for each 'question_excluding_window_X' column
# for col in df.columns:
#     if col.startswith("question_excluding_window_"):
#         df[col + "_vector"] = df[col].apply(get_sentence_vector)
#
# # Save the updated dataset
# updated_file_path = "../data/strategyqa_question_vectors.csv"
# df.to_csv(updated_file_path, index=False)
#
# print("\nDataset successfully updated with question embeddings and saved as 'strategyqa_question_vectors.csv'!")


# # Load the dataset with embeddings
# file_path = "../data/strategyqa_question_vectors.csv"
# df = pd.read_csv(file_path)
#
# # Convert string representations of lists back to actual lists
# df["question_vector"] = df["question_vector"].apply(ast.literal_eval)
#
# # Identify columns with embeddings for excluded windows
# for col in df.columns:
#     if col.startswith("question_excluding_window_") and col.endswith("_vector"):
#         df[col] = df[col].apply(ast.literal_eval)  # Convert string to list
#         similarity_col_name = col.replace("_vector", "_cosine_similarity")
#
#         # Compute cosine similarity
#         df[similarity_col_name] = df.apply(
#             lambda row: cosine_similarity([row["question_vector"]], [row[col]])[0][0]
#             if isinstance(row["question_vector"], list) and isinstance(row[col], list) else np.nan,
#             axis=1
#         )
#
# # Save the updated dataset with cosine similarities
# similarity_file_path = "../data/strategyqa_cosine_similarity.csv"
# df.to_csv(similarity_file_path, index=False)
#
# print("\nDataset successfully updated with cosine similarities and saved as 'strategyqa_cosine_similarity.csv'!")





# import os
# import pandas as pd
# import nltk
# import numpy as np
# import ast
# from nltk.corpus import stopwords
# from sentence_transformers import SentenceTransformer
#
# # Download stopwords if not already present
# nltk.download('stopwords')
# stop_words = set(stopwords.words("english"))
#
# # Initialize sentence embedding model
# model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# instruction = "Generate a representation for this sentence to retrieve related articles: "
#
# # Set paths
# raw_data_folder = "../data/raw_datasets"
# processed_data_folder = "../data/processed_datasets"
#
# os.makedirs(processed_data_folder, exist_ok=True)
#
# # Function to remove stopwords
# def remove_stopwords(text):
#     if isinstance(text, str):
#         words = text.split()
#         filtered = [w for w in words if w.lower() not in stop_words]
#         return " ".join(filtered)
#     return ""
#
# # Sliding window function
# def sliding_window(text, window_size=3):
#     words = text.split()
#     return [" ".join(words[i:i+window_size]) for i in range(0, len(words), window_size)] or [text]
#
# # Function to generate embedding
# def get_embedding(text):
#     if not isinstance(text, str):
#         text = ""
#     return model.encode(instruction + text, normalize_embeddings=True).tolist()
#
# # Process a single dataset
# def process_dataset(file_path):
#     df = pd.read_csv(file_path)
#
#     # Standardize column name (if needed)
#     if "question" not in df.columns:
#         possible_cols = [col for col in df.columns if "question" in col.lower()]
#         if possible_cols:
#             df.rename(columns={possible_cols[0]: "question"}, inplace=True)
#         else:
#             raise ValueError(f"No 'question' column found in {file_path}")
#
#     # Remove stopwords
#     df["question_no_stopwords"] = df["question"].apply(remove_stopwords)
#
#     # Apply sliding windows
#     df["question_windows"] = df["question_no_stopwords"].apply(sliding_window)
#
#     # Exclude each window and create variants
#     def remove_each_window(question, windows):
#         variants = {}
#         for i in range(len(windows)):
#             remaining = windows[:i] + windows[i+1:]
#             variants[f"question_excluding_window_{i+1}"] = " ".join(remaining)
#         return variants
#
#     variants = df.apply(lambda row: remove_each_window(row["question_no_stopwords"], row["question_windows"]), axis=1)
#     variants_df = pd.DataFrame(variants.tolist())
#     df = pd.concat([df, variants_df], axis=1)
#
#     # Generate embeddings
#     df["question_vector"] = df["question_no_stopwords"].apply(get_embedding)
#
#     for col in variants_df.columns:
#         df[col + "_vector"] = df[col].apply(get_embedding)
#
#     # Add placeholder label
#     if "contains_trigger" not in df.columns:
#         df["contains_trigger"] = 0
#
#     # Save processed file
#     file_name = os.path.basename(file_path).replace("_dataset.csv", "_processed.csv")
#     save_path = os.path.join(processed_data_folder, file_name)
#     df.to_csv(save_path, index=False)
#     print(f"Processed and saved: {save_path}")
#
# # Process all files
# for file_name in os.listdir(raw_data_folder):
#     if file_name.endswith(".csv"):
#         process_dataset(os.path.join(raw_data_folder, file_name))
#
# print("\nAll datasets have been preprocessed and saved!")






import os
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

# Setup NLTK and model
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
instruction = "Generate a representation for this sentence to retrieve related articles: "

# Paths
raw_data_folder = "../data/raw_datasets"
processed_data_folder = "../data/processed_datasets"
os.makedirs(processed_data_folder, exist_ok=True)

# Remove stopwords
def remove_stopwords(text):
    if isinstance(text, str):
        words = text.split()
        return " ".join([w for w in words if w.lower() not in stop_words])
    return ""

# Sliding window
def sliding_window(text, window_size=3):
    words = text.split()
    return [" ".join(words[i:i+window_size]) for i in range(0, len(words), window_size)] or [text]

# Process single file
def process_dataset(file_path):
    print(f"\nProcessing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)

    # Ensure question column exists
    if "question" not in df.columns:
        possible = [col for col in df.columns if "question" in col.lower()]
        if possible:
            df.rename(columns={possible[0]: "question"}, inplace=True)
        else:
            raise ValueError(f"No question column in {file_path}")

    # Step 1: Remove stopwords
    df["question_no_stopwords"] = df["question"].apply(remove_stopwords)

    # Step 2: Sliding windows
    df["question_windows"] = df["question_no_stopwords"].apply(sliding_window)

    # Step 3: Create window-excluded variants
    def remove_each_window(windows):
        return {f"question_excluding_window_{i+1}": " ".join(windows[:i] + windows[i+1:])
                for i in range(len(windows))}
    variant_dicts = df["question_windows"].apply(remove_each_window)
    variants_df = pd.DataFrame(variant_dicts.tolist())
    df = pd.concat([df, variants_df], axis=1)

    # Step 4: Gather all sentences to embed
    all_text_cols = ["question_no_stopwords"] + list(variants_df.columns)
    all_sentences = []
    for col in all_text_cols:
        all_sentences.extend(df[col].fillna("").astype(str).tolist())

    print(f"Encoding {len(all_sentences)} sentences...")
    all_embeddings = model.encode(
        [instruction + s for s in all_sentences],
        batch_size=32,
        normalize_embeddings=True
    )

    # Step 5: Split embeddings back into columns
    num_rows = len(df)
    chunk_size = num_rows
    chunked_embeddings = [all_embeddings[i*chunk_size:(i+1)*chunk_size] for i in range(len(all_text_cols))]

    df["question_vector"] = chunked_embeddings[0].tolist()
    for i, col in enumerate(variants_df.columns):
        df[col + "_vector"] = chunked_embeddings[i + 1].tolist()

    # # Step 6: Add label
    # if "contains_trigger" not in df.columns:
    #     df["contains_trigger"] = 0

    # Save
    file_name = os.path.basename(file_path).replace("_dataset.csv", "_processed.csv")
    output_path = os.path.join(processed_data_folder, file_name)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# Run on all datasets
for file in os.listdir(raw_data_folder):
    if file.endswith(".csv"):
        process_dataset(os.path.join(raw_data_folder, file))

print("\nAll datasets preprocessed and saved!")
