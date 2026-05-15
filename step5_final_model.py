import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 1. DATA LOADING & PREPROCESSING
# Ensure the CSV is in the same folder as this script
df = pd.read_csv('data/transactions_upi_transaction_failure.csv')

# Feature Engineering: Extracting patterns relevant to rural environments
df['Bank'] = df['Sender UPI ID'].str.split('@').str[1]
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

# Feature Selection: Choosing variables that impact transaction success
X = df[['Amount (INR)', 'Bank', 'Hour', 'Day_of_Week']]
y = df['Status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)

# Encoding: Converting categorical bank names into numerical values for the ML math
le = LabelEncoder()
X.loc[:, 'Bank'] = le.fit_transform(X['Bank'])

# 2. DATA SPLITTING
# Training on 80% of data, Testing on 20% to validate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. MODEL TRAINING
print("Training the Random Forest Classifier...")
# Random Forest is an ensemble of decision trees, ideal for tabular data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 4. PREDICTION & EVALUATION
y_pred = rf_model.predict(X_test)

print("\n--- MODEL PERFORMANCE ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 5. FEATURE IMPORTANCE (Updated Bar Plot Code)
importances = rf_model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
# Updated barplot parameters to avoid FutureWarning
sns.barplot(
    data=importance_df,
    x='Importance',
    y='Feature',
    hue='Feature',
    palette='viridis',
    legend=False
)
plt.title('Feature Importance: Which Factors Predict UPI Failure Most?')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

# 6. CONFUSION MATRIX (Visualizing Accuracy)
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Failed', 'Success'], yticklabels=['Failed', 'Success'])
plt.xlabel('Predicted Status')
plt.ylabel('Actual Status')
plt.title('Confusion Matrix: Transaction Success vs Failure')
plt.show()