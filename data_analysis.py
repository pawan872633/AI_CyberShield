import pandas as pd

# ✅ Apni file ka path yaha likh
file_path = r"C:\Users\pawan\Downloads\GeneratedLabelledFlows\TrafficLabelling\Monday-WorkingHours.pcap_ISCX.csv"

# ✅ CSV file load karna
df = pd.read_csv(file_path)

# ✅ Pehli 5 rows print karna
print("🔹 First 5 Rows of Data:")
print(df.head())

# ✅ Data ka structure dekhna
print("\n🔹 Dataset Structure:")
print(df.info())

# ✅ Missing values check karna
print("\n🔹 Missing Values in Each Column:")
print(df.isnull().sum())
     