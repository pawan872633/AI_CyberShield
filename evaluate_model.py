import pandas as pd
import joblib
from sklearn.metrics import classification_report

# ✅ Model, scaler, and encoders load karna
try:
    model = joblib.load('model/best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('label_encoders.pkl')  # Dictionary of encoders
    print("✅ Model, Scaler, and Encoders Loaded Successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    exit()

# ✅ Test data load karna
test_data = pd.read_csv("dataset/intrusion_data.csv")

# ✅ Target column identify karna
target_column = "attack_type"  # Change if different in dataset

# ✅ Features aur labels separate karna
X_test = test_data.drop(target_column, axis=1)
y_test = test_data[target_column]

# ✅ Encode target labels (y_test)
if target_column in encoders:
    y_test = encoders[target_column].transform(y_test)  # 🛠 FIXED: Encode `y_test`

# ✅ Categorical columns encode karna
for col in X_test.select_dtypes(include=['object']).columns:
    if col in encoders:
        X_test[col] = encoders[col].transform(X_test[col])

# ✅ Feature scaling apply karna
X_test_scaled = scaler.transform(X_test)

# ✅ Prediction
y_pred = model.predict(X_test_scaled)

# ✅ Classification report print karna
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Model Evaluation Completed Successfully!")

