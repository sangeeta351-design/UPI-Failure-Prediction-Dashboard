import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
# Make sure the file path matches where you saved your CSV
try:
    df = pd.read_csv('data/transactions_upi_transaction_failure.csv')
except FileNotFoundError:
    df = pd.read_csv('transactions_upi_transaction_failure.csv')

# 2. Extract 'Rural-Focus' Features for Analysis
# Extract Bank name from UPI ID (e.g., oksbi, okaxis)
df['Bank'] = df['Sender UPI ID'].str.split('@').str[1]

# Convert Timestamp to datetime and extract the Hour
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour

# 3. Visualizations
plt.figure(figsize=(15, 10))

# Plot A: Overall Success vs Failure
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='Status', palette='viridis')
plt.title('Overall Transaction Status')

# Plot B: Performance by Bank
# This is crucial for rural analysis as some banks have better rural reach
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Bank', hue='Status', palette='magma')
plt.title('Status Distribution per Bank')
plt.xticks(rotation=45)

# Plot C: Failures by Time (Hour of Day)
# Rural network congestion often happens at specific hours
plt.subplot(2, 2, 3)
sns.countplot(data=df, x='Hour', hue='Status', palette='coolwarm')
plt.title('Transactions by Hour of Day')

# Plot D: Amount vs Failure
# Do larger amounts fail more in rural areas?
plt.subplot(2, 2, 4)
sns.boxplot(data=df, x='Status', y='Amount (INR)', palette='Set2')
plt.title('Amount Distribution (Success vs Failure)')

plt.tight_layout()
plt.show()  # This will pop up the window with charts in PyCharm
