import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 📌 Data Load
try:
    data = pd.read_csv("dataset/intrusion_data.csv")
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Dataset file not found! Check the path.")
    exit()

# 📌 Automatically detect target column (assuming last column is the target)
target_column = data.columns[-1]  # Last column is assumed as the target
print(f"🎯 Detected target column: {target_column}")

# 📌 Convert categorical features to numeric
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
print(f"🔄 Converting categorical columns: {categorical_columns}")

label_encoders = {}  # Store encoders for future use
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store encoder

# 📌 Features & Labels
X = data.drop(target_column, axis=1)
y = data[target_column]

# 📌 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 📌 Model Training (Balanced RandomForest)
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# 📌 Predictions
y_pred = model.predict(X_test_scaled)

# 📌 Evaluation
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 📌 Save Model, Scaler & Label Encoders
joblib.dump(model, "final_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")  # Save encoders for future decoding
print("✅ Model, Scaler & Encoders saved successfully!")



