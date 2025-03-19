import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("🚀 Scaler Training Started...\n")

# ✅ Dataset Load Karo
try:
    df = pd.read_csv("dataset/intrusion_data.csv")  # Dataset Load Karo
    print("✅ Dataset Loaded Successfully!\n")
except Exception as e:
    print(f"❌ Error Loading Dataset: {e}")
    exit()

# ✅ Features Define Karo (🔥 'attack_type' HATA DIYA)
feature_columns = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "packet_size", "connection_duration"]

# ✅ Missing Features Check Karo
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"❌ Missing Features in Dataset: {missing_features}")
    exit()

# ✅ 'protocol' Column Encode Karo (🔥 Fix for String Values)
if df["protocol"].dtype == "object":
    le = LabelEncoder()
    df["protocol"] = le.fit_transform(df["protocol"])  # Convert Protocol to Numeric
    joblib.dump(le, "protocol_encoder.pkl")  # Save Encoder for Future Use
    print("✅ Protocol Encoding Completed & Saved!\n")

# ✅ Feature Data Extract Karo
X = df[feature_columns]

# ✅ Scaler Train Karo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Scaler Save Karo
joblib.dump(scaler, "scaler.pkl")
print(f"✅ Scaler trained & saved successfully on features: {feature_columns}\n")

print("🚀 Scaler Training Completed Successfully! ✅")
