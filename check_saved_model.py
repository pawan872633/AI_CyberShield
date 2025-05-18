import joblib

try:
    # Load best model
    best_model = joblib.load("best_model.pkl")
    print("✅ Best Model Loaded Successfully!")

    # Load scaler
    scaler = joblib.load("scaler.pkl")
    print(f"✅ Scaler Loaded! Type: {type(scaler)}")

    # Check if scaler is StandardScaler
    if hasattr(scaler, "mean_"):
        print(f"✅ Scaler Mean: {scaler.mean_}")

except FileNotFoundError:
    print("❌ Model ya Scaler file nahi mili! Pehle train karna padega.")
except Exception as e:
    print(f"❌ Error: {e}")

