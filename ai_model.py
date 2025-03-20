import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# ✅ Load Dataset
try:
    df = pd.read_csv("dataset/intrusion_data.csv")
    print("✅ Dataset Loaded Successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset File Not Found! Check Path.")
    exit()

# ✅ Ensure Required Columns Exist
required_columns = ["src_port", "dst_port", "packet_size", "connection_duration", "protocol", "attack_type"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"❌ Error: Missing columns in dataset -> {missing_columns}")
    exit()

# ✅ Label Encoding for 'protocol'
protocol_encoder = LabelEncoder()
df["protocol"] = protocol_encoder.fit_transform(df["protocol"])
with open("protocol_encoder.pkl", "wb") as f:
    pickle.dump(protocol_encoder, f)
print("✅ Protocol Encoder Saved Successfully!")

# ✅ Label Encoding for 'attack_type'
attack_encoder = LabelEncoder()
df["attack_type"] = attack_encoder.fit_transform(df["attack_type"])
with open("attack_encoder.pkl", "wb") as f:
    pickle.dump(attack_encoder, f)
print("✅ Attack Type Encoder Saved Successfully!")

# ✅ Feature Selection
X = df.drop(columns=["attack_type"])
y = df["attack_type"]
print(f"🔍 Features Before SMOTE: {X.shape[1]} Columns")

# ✅ Handle Data Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("🔍 Class Distribution After SMOTE:", Counter(y_resampled))

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# ✅ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler Saved Successfully!")

# ✅ Train Model (Hyperparameter Tuning for Accuracy)
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=20, 
    min_samples_split=5, 
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ✅ Save Model
with open("best_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ AI Model Trained & Saved Successfully! 🚀")

# ✅ Feature Importance Debugging
importances = model.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"⭐ {feature}: {importance:.4f}")

# ✅ Debug: Ensure Feature Count Matches
print(f"✅ Model Trained with {X_train.shape[1]} Features")

