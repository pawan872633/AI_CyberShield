import pandas as pd
import numpy as np

# Sample Intrusion Detection Dataset
np.random.seed(42)
n_samples = 1000  # Number of data points

# Feature columns (e.g., network traffic attributes)
data = {
    "src_ip": np.random.randint(1, 255, n_samples),
    "dst_ip": np.random.randint(1, 255, n_samples),
    "src_port": np.random.randint(1024, 65535, n_samples),
    "dst_port": np.random.randint(1, 65535, n_samples),
    "protocol": np.random.choice(["TCP", "UDP", "ICMP"], n_samples),
    "packet_size": np.random.randint(64, 1500, n_samples),
    "connection_duration": np.random.uniform(0.1, 10, n_samples),
    "attack_type": np.random.choice(["normal", "dos", "r2l", "u2r", "probe"], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset to CSV
df.to_csv("dataset/intrusion_data.csv", index=False)

print("✅ Dataset 'intrusion_data.csv' successfully created in 'dataset' folder!")
