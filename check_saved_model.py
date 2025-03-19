import joblib

try:
    # ✅ Load saved model
    best_model = joblib.load("best_model.pkl")
    print("✅ Best Model Loaded Successfully!")

    # ✅ Load scaler
    scaler_data = joblib.load("scaler.pkl")
    scaler = scaler_data["scaler"]
    feature_names = scaler_data["features"]

    print(f"✅ Scaler Loaded! Features used: {feature_names}")

except FileNotFoundError:
    print("❌ Model ya Scaler file nahi mili! Pehle train karna padega.")
except Exception as e:
    print(f"❌ Error: {e}")
