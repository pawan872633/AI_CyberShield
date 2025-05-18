import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# ✅ Correct feature names
feature_names = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'packet_size', 'connection_duration']

# ✅ New test sample (with correct feature names)
sample = pd.DataFrame([[150, 130, 35000, 34500, 0, 800, 7]], columns=feature_names)

# Transform and predict
scaled_sample = scaler.transform(sample)
prediction = model.predict(scaled_sample)

print("🔍 New Prediction:", prediction)
