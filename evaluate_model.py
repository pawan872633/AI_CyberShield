from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np

# ✅ Model & Scaler Load
model = pickle.load(open("best_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("🚀 Model & Scaler Loaded Successfully!")

# ✅ New Test Data (More Attack Scenarios)
X_test = np.array([
    [80, 443, 200, 1.5, 0.8, 0.75, 6, 0.5],    # Normal Packet (0)
    [5000, 22, 1500, 5.2, 0.3, 0.5, 17, 2.0],  # Possible Attack (3)
    [53, 53, 500, 1.2, 0.9, 0.85, 6, 0.3],     # DNS Traffic (1)
    [7000, 80, 2500, 7.0, 0.1, 0.6, 20, 3.0],  # High Traffic Load (2)
    [0, 0, 0, 0.0, 0.0, 0.0, 0, 0.0],          # Zero Traffic (Edge Case)
    [3000, 443, 1200, 4.5, 0.4, 0.55, 15, 1.8] # Unknown Traffic Type (New Case)
])

# ✅ True Labels (Manually Defined)
y_true = [0, 3, 1, 2, 0, 2]  # Adjusted labels with new cases

# ✅ Scale Test Data
X_test_scaled = scaler.transform(X_test)

# ✅ Predict
y_pred = model.predict(X_test_scaled)

# ✅ Accuracy Report
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred))
