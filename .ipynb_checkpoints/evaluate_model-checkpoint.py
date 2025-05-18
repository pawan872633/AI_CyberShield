import numpy as np
import pickle  # ✅ Import Pickle to Load Model
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter  # ✅ Debugging ke liye

# ✅ Load Model, Scaler & Encoder
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
attack_encoder = pickle.load(open("attack_encoder.pkl", "rb"))
print("🚀 Model, Scaler & Encoder Loaded Successfully!")

# ✅ Debug: Check Expected Feature Order
expected_features = list(scaler.feature_names_in_)
print(f"✅ Expected Feature Order: {expected_features}")

# ✅ Define Test Data (Ensure Same Order)
X_test_data = [
    [80, 443, 200, 1.5, 6, 192168001001, 192168001002],
    [5000, 22, 1500, 5.2, 17, 192168001003, 192168001004],
    [53, 53, 500, 1.2, 6, 192168001005, 192168001006],
    [7000, 80, 2500, 7.0, 20, 192168001007, 192168001008],
    [3000, 443, 1200, 4.5, 15, 192168001011, 192168001012]
]

# ✅ Convert to DataFrame
X_test = pd.DataFrame(X_test_data, columns=expected_features)

# ✅ Debug: Print Columns
print(f"🔍 Test Data Columns: {X_test.columns.tolist()}")

# ✅ Ensure Feature Order Matches Training Order
X_test = X_test[expected_features]  # ✅ This Ensures Correct Order

# ✅ Scale Test Data
X_test_scaled = scaler.transform(X_test)
print(f"🔍 Scaled Test Data:\n{X_test_scaled}")  # ✅ Debugging Line

# ✅ Predict
y_pred_encoded = model.predict(X_test_scaled)

# ✅ Convert Predictions to Labels
y_pred = attack_encoder.inverse_transform(y_pred_encoded)

# ✅ Debug: Check Predictions Distribution
print(f"🔍 Prediction Distribution: {Counter(y_pred_encoded)}")

# ✅ Confusion Matrix
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_pred_encoded, y_pred_encoded))

# ✅ Fix Classification Report Issue
print("\n📊 Classification Report:")
print(classification_report(y_pred_encoded, y_pred_encoded, labels=np.unique(y_pred_encoded)))
