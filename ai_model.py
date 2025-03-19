import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# ✅ Load dataset
df = pd.read_csv("dataset/intrusion_data.csv")
print("✅ Dataset loaded successfully!")

# ✅ Encode categorical columns
if 'protocol' in df.columns:
    df['protocol'] = df['protocol'].astype('category').cat.codes  

# ✅ Encode target column
target_col = "attack_type"
df[target_col] = df[target_col].astype("category").cat.codes

# ✅ Drop unnecessary features
df.drop(columns=[col for col in ['src_ip', 'dst_ip'] if col in df.columns], inplace=True)

# ✅ Feature Engineering & Normalization
df["packet_size"] = np.log1p(df["packet_size"].clip(lower=1))  
if 'bytes_sent' in df.columns and 'bytes_received' in df.columns:
    df["bytes_ratio"] = np.log1p(df["bytes_sent"] / (df["bytes_received"] + 1))  

df["packet_ratio"] = np.log1p(df["packet_size"] / (df["src_port"].replace(0, 1) + df["dst_port"].replace(0, 1)))
df["port_ratio"] = np.log1p(df["src_port"] / df["dst_port"].replace(0, 1))
df["src_port_dst_port"] = np.log1p(df["src_port"] * df["dst_port"])

# ✅ Selecting important features
feature_cols = ['src_port', 'dst_port', 'packet_size', 'packet_ratio', 'port_ratio', 'src_port_dst_port']
if 'protocol' in df.columns:
    feature_cols.append('protocol')
if 'connection_duration' in df.columns:
    feature_cols.append('connection_duration')
if 'bytes_ratio' in df.columns:
    feature_cols.append('bytes_ratio')

X = df[feature_cols]
y = df[target_col]

# ✅ Handling class imbalance using SMOTE
class_counts = Counter(y)
sampling_strategy = {cls: max(class_counts.values()) for cls in class_counts}  # Balance to majority class
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ✅ Feature Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_resampled)

# ✅ Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# ✅ Hyperparameter tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [10, 20, 30], 
    'min_samples_split': [2, 5, 10]
}
rf_model = GridSearchCV(RandomForestClassifier(class_weight='balanced_subsample'), rf_params, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
rf_model.fit(X_train, y_train)
best_rf_params = rf_model.best_params_
print(f"🔍 Best RF Params: {best_rf_params}")

# ✅ Hyperparameter tuning for XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300], 
    'max_depth': [5, 10, 15], 
    'learning_rate': [0.01, 0.05, 0.1],
    'scale_pos_weight': [1, 2, 5]
}
xgb_model = GridSearchCV(XGBClassifier(eval_metric='mlogloss'), xgb_params, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
xgb_model.fit(X_train, y_train)
best_xgb_params = xgb_model.best_params_
print(f"🔍 Best XGB Params: {best_xgb_params}")

# ✅ Train final models
rf_final = RandomForestClassifier(**best_rf_params, class_weight='balanced_subsample')
rf_final.fit(X_train, y_train)

xgb_final = XGBClassifier(**best_xgb_params, eval_metric='mlogloss')
xgb_final.fit(X_train, y_train)

# ✅ Model Evaluation
rf_pred = rf_final.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_final.predict_proba(X_test), multi_class='ovr')

xgb_pred = xgb_final.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
xgb_auc = roc_auc_score(y_test, xgb_final.predict_proba(X_test), multi_class='ovr')

print(f"✅ Random Forest Accuracy: {rf_acc:.2f}, AUC: {rf_auc:.2f}")
print("📜 RF Classification Report:\n", classification_report(y_test, rf_pred))

print(f"✅ XGBoost Accuracy: {xgb_acc:.2f}, AUC: {xgb_auc:.2f}")
print("📜 XGB Classification Report:\n", classification_report(y_test, xgb_pred))

# ✅ Confusion Matrices
print("🔍 RF Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("🔍 XGB Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

# ✅ Save Best Model
best_model = rf_final if rf_auc > xgb_auc else xgb_final
best_model_name = "Random Forest" if rf_auc > xgb_auc else "XGBoost"
joblib.dump(best_model, "best_model.pkl")

# ✅ Save Scaler with Feature Names
joblib.dump({"scaler": scaler, "features": feature_cols}, "scaler.pkl")

print(f"🏆 Best Model Selected: {best_model_name} and saved successfully!")
