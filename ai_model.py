import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ✅ Step 1: Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("dataset/intrusion_data.csv")  # ✅ Sahi dataset ka naam

print("✅ Dataset loaded successfully!")

# ✅ Step 2: Encode categorical features
print("🔄 Encoding categorical features...")
df = pd.get_dummies(df, drop_first=True)
print("✅ Encoding completed!")

# ✅ Step 3: Feature Engineering
print("🛠 Feature Engineering: Creating new features...")
df["packet_ratio"] = df["packet_size"] / (df["src_port"] + df["dst_port"] + 1)
df["src_port_dst_port"] = df["src_port"] * df["dst_port"]
print("✅ New features added!")

# ✅ Step 4: Remove constant features
df = df.loc[:, (df != df.iloc[0]).any()]
print("✅ Removed constant features!")

# ✅ Step 5: Select best features
print("🧠 Selecting best features...")
feature_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'packet_size', 'packet_ratio', 'src_port_dst_port']
X = df[feature_cols]
y = df["attack_type"]  # 🔹 "label" ki jagah "attack_type" use kar

print(f"✅ Selected Features: {feature_cols}")

# ✅ Step 6: Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Feature scaling (MinMax) completed!")

# ✅ Step 7: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("✅ Dataset split completed!")

# ✅ Step 8: Hyperparameter tuning
print("🚀 Hyperparameter tuning for Random Forest...")
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20], 'min_samples_split': [2, 5, 10]}
rf_grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf_params = rf_grid.best_params_  # ✅ Save best RF parameters
print(f"✅ Best Random Forest Model: {best_rf_params}")

print("🚀 Hyperparameter tuning for XGBoost...")
param_grid_xgb = {'n_estimators': [100, 200], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.1]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5, scoring='accuracy')
xgb_grid.fit(X_train, y_train)
best_xgb_params = xgb_grid.best_params_  # ✅ Save best XGB parameters
print(f"✅ Best XGBoost Model: {best_xgb_params}")

# ✅ Step 9: Train best models
print("🚀 Training the best Random Forest and XGBoost models...")
rf_model = RandomForestClassifier(**best_rf_params)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(**best_xgb_params)
xgb_model.fit(X_train, y_train)
print("✅ Model training completed!")

# ✅ Step 10: Model Evaluation
print("📊 Evaluating models...")
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"✅ Random Forest Accuracy: {rf_accuracy:.2f}")
print("📜 RF Classification Report:\n", classification_report(y_test, rf_pred))

xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"✅ XGBoost Accuracy: {xgb_accuracy:.2f}")
print("📜 XGB Classification Report:\n", classification_report(y_test, xgb_pred))

# ✅ Step 11: Save Best Model
best_model = rf_model if rf_accuracy > xgb_accuracy else xgb_model
best_model_name = "Random Forest" if rf_accuracy > xgb_accuracy else "XGBoost"
print(f"🏆 Best Model Selected: {best_model_name}")

print("💾 Saving the best model...")
import joblib
joblib.dump(best_model, "best_model.pkl")
print("✅ Best model saved successfully!")


