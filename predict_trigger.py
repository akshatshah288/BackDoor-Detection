# import pandas as pd
# import torch
# from train_model import TriggerNet
# from utils import load_features
# from config import MODEL_PATH, DATA_FILE, PREDICTION_OUTPUT
#
# # Load data
# df = pd.read_csv(DATA_FILE)
# X = load_features(df)
#
# # Load model
# model = TriggerNet(input_size=X.shape[1])
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()
#
# # Predict
# with torch.no_grad():
#     inputs = torch.tensor(X, dtype=torch.float32)
#     outputs = model(inputs).squeeze()
#     predictions = (outputs > 0.5).int().numpy()
#
# # Add predictions to dataframe
# df["predicted_trigger"] = predictions
#
# # Save to file
# df.to_csv(PREDICTION_OUTPUT, index=False)
# print(f"Predictions saved to {PREDICTION_OUTPUT}")



import pandas as pd
import torch
from train_model import TriggerNet
from utils import load_features
from config import MODEL_PATH, DATA_FILE, PREDICTION_OUTPUT

# Load dataset
df = pd.read_csv(DATA_FILE)

# Load features from vector columns
X = load_features(df)
inputs = torch.tensor(X, dtype=torch.float32)

# Load trained model
model = TriggerNet(input_size=X.shape[1])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(inputs).squeeze()
    predictions = (outputs > 0.5).int().numpy()

# Add predictions to the DataFrame
df["predicted_trigger"] = predictions

# Save output file
df.to_csv(PREDICTION_OUTPUT, index=False)
print(f"Predictions saved to {PREDICTION_OUTPUT}")
