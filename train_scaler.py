import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ✅ Dataset Load Karo
df = pd.read_csv("dataset/intrusion_data.csv")  # Fix Path

# ✅ Categorical Columns Encode Karo (Agar Hai)
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])  # Encode Karo
    label_encoders[column] = le

# ✅ Feature Selection (Assuming Last Column = Target)
X = df.iloc[:, :-1]  # Last column ko hata diya

# ✅ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save Scaler & Encoders
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")  # Save Encoders

print("✅ Scaler & Label Encoders Saved Successfully! 🚀")
