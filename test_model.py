import joblib
import numpy as np

print("🚀 Model Testing Started...\n")

# ✅ Model & Scaler Load Karo
try:
    model = joblib.load("best_model.pkl")  # Model Load
    scaler = joblib.load("scaler.pkl")  # Scaler Load
    print("✅ Model & Scaler Loaded Successfully!\n")
except Exception as e:
    print(f"❌ Error Loading Model/Scaler: {e}")
    exit()

# ✅ Debugging: Scaler Ke Expected Features Check Karo
if hasattr(scaler, "feature_names_in_"):
    print(f"🔍 Scaler Expected Features: {scaler.feature_names_in_}\n")
    expected_features = len(scaler.feature_names_in_)
else:
    print("⚠️ Scaler does not have 'feature_names_in_' attribute. Defaulting to 8 features.\n")
    expected_features = 8  # Default value

# ✅ Test Data Define Karo (🔥 Fix: Ensure Correct Number of Features)
test_cases = {
    "Normal Packet": [[19216801, 19216802, 80, 443, 6, 200, 1.5, 0][:expected_features]],  # Added 8th Feature
    "Possible Attack": [[19216810, 19216820, 5000, 22, 17, 1500, 5.2, 1][:expected_features]],  # Added 8th Feature
    "DNS Traffic": [[19216830, 19216840, 53, 53, 17, 500, 1.2, 0][:expected_features]],  # Added 8th Feature
    "High Traffic Load": [[19216850, 19216860, 7000, 80, 6, 2500, 7.0, 1][:expected_features]]  # Added 8th Feature
}

# ✅ Model Prediction Karo
for case, data in test_cases.items():
    try:
        scaled_data = scaler.transform(data)  # Data Normalize Karo
        prediction = model.predict(scaled_data)  # Prediction Karo
        print(f"🔍 {case} -> 🔥 Predicted Attack Type: {prediction[0]}")
    except Exception as e:
        print(f"❌ Error Testing '{case}': {e}")

print("\n✅ Model Testing Completed Successfully! 🚀")
