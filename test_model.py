import joblib
import numpy as np
import pandas as pd  # ✅ Fix ke liye DataFrame ka use

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
    expected_features = scaler.feature_names_in_
else:
    print("⚠️ Scaler does not have 'feature_names_in_' attribute. Defaulting to standard features.\n")
    expected_features = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_size', 'connection_duration']

# ✅ Test Data Define Karo (🔥 Fix: Ensure Correct Feature Names)
test_cases = {
    "Normal Packet": [19216801, 19216802, 80, 443, 6, 200, 1.5],
    "Possible Attack": [19216810, 19216820, 5000, 22, 17, 1500, 5.2],
    "DNS Traffic": [19216830, 19216840, 53, 53, 17, 500, 1.2],
    "High Traffic Load": [19216850, 19216860, 7000, 80, 6, 2500, 7.0]
}

# ✅ Model Prediction Karo
for case, data in test_cases.items():
    try:
        # 🔥 Convert to DataFrame with Correct Feature Names
        test_df = pd.DataFrame([data], columns=expected_features)
        
        # ✅ Apply Scaler
        scaled_data = scaler.transform(test_df)

        # ✅ Prediction Karo
        prediction = model.predict(scaled_data)
        print(f"🔍 {case} -> 🔥 Predicted Attack Type: {prediction[0]}")
    except Exception as e:
        print(f"❌ Error Testing '{case}': {e}")

print("\n✅ Model Testing Completed Successfully! 🚀")

