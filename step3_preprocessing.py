import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the data
df = pd.read_csv('data/transactions_upi_transaction_failure.csv')
# 2. Feature Engineering (Repeating the extraction from Step 2)
df['Bank'] = df['Sender UPI ID'].str.split('@').str[1]
df['Hour'] = pd.to_datetime(df['Timestamp']).dt.hour
df['Day_of_Week'] = pd.to_datetime(df['Timestamp']).dt.dayofweek

# 3. Selecting the Features (X) and the Target (y)
# We only keep things that have a pattern: Amount, Bank, and Time.
X = df[['Amount (INR)', 'Bank', 'Hour', 'Day_of_Week']]
y = df['Status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)

# 4. Encoding the 'Bank' column into numbers
# Machine Learning models can't read "oksbi", they need numbers like 0, 1, 2...
le = LabelEncoder()
X['Bank'] = le.fit_transform(X['Bank'])

# 5. Splitting the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Preprocessing Complete!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print("\nFirst 5 rows of processed features:")
print(X_train.head())