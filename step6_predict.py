import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. DATA LOADING & MODEL TRAINING ---
# Ensure 'transactions_upi_transaction_failure.csv' is in the same directory
df = pd.read_csv('data/transactions_upi_transaction_failure.csv')

# Feature Engineering: Extract attributes representing rural transactional environment
df['Bank'] = df['Sender UPI ID'].str.split('@').str[1]
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

# Select structural predictive features and binary status target
X = df[['Amount (INR)', 'Bank', 'Hour', 'Day_of_Week']]
y = df['Status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)

# Label Encoding for categorical bank attributes
le = LabelEncoder()
X.loc[:, 'Bank'] = le.fit_transform(X['Bank'])

# Split data and fit the Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# --- 2. THE PREDICTION ENGINE ---
def predict_new_transaction(amount, bank_name, hour, day_of_week):
    """
    Function to evaluate and predict safety/failure metrics of a single transaction vector.
    day_of_week index: 0=Monday, 1=Tuesday, ..., 6=Sunday
    """
    try:
        # Transform the typed bank string into its mapped internal numeric label
        encoded_bank = le.transform([bank_name])[0]
    except ValueError:
        print(f"❌ Error: Bank Handle '{bank_name}' not recognized. Choose from: {list(le.classes_)}")
        return

    # Structure user input values into a single row dataframe matching training features
    custom_data = pd.DataFrame([{
        'Amount (INR)': amount,
        'Bank': encoded_bank,
        'Hour': hour,
        'Day_of_Week': day_of_week
    }])

    # Predict status classification (0 or 1)
    prediction = model.predict(custom_data)[0]

    # Extract prediction probability score arrays
    probabilities = model.predict_proba(custom_data)[0]
    fail_prob = probabilities[0] * 100
    success_prob = probabilities[1] * 100

    print("\n" + "=" * 45)
    print("          UPI TRANSACTION EVALUATION          ")
    print("=" * 45)
    print(f"Inputs -> Amount: ₹{amount} | Bank: {bank_name} | Time: {hour}:00 | Day: {day_of_week}")
    print("---------------------------------------------")
    if prediction == 1:
        print(f"Prediction: SUCCESS ✅ (Confidence: {success_prob:.1f}%)")
    else:
        print(f"Prediction: FAILED ❌ (Confidence: {fail_prob:.1f}%)")
    print("=" * 45 + "\n")


# --- 3. INTERACTIVE TERMINAL MENU FOR EXAMINERS ---
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("          UPI FAILURE PREDICTION SYSTEM ONLINE          ")
    print("============================================================")
    print(f"Available Banks in system: {list(le.classes_)}")
    print("Days mapping: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun")
    print("=" * 60)

    while True:
        try:
            print("\nEnter Transaction Details (or type Ctrl+C to exit):")

            # Capture inputs interactively from terminal console
            user_amount = float(input("1. Enter Amount in INR (e.g., 500): "))
            user_bank = input("2. Enter Bank Handle (e.g., oksbi, okaxis): ").strip().lower()
            user_hour = int(input("3. Enter Hour of Day (0-23): "))
            user_day = int(input("4. Enter Day Index (0-6): "))

            # Validate numeric bounds
            if not (0 <= user_hour <= 23) or not (0 <= user_day <= 6):
                print("❌ Invalid input bounds! Hour must be 0-23 and Day index must be 0-6.")
                continue

            # Feed input directly into predictor engine
            predict_new_transaction(
                amount=user_amount,
                bank_name=user_bank,
                hour=user_hour,
                day_of_week=user_day
            )

            # Continuous testing verification check
            cont = input("Test another transaction? (y/n): ").strip().lower()
            if cont != 'y':
                print("\nExiting system. Good luck with your project evaluation!")
                break

        except ValueError:
            print("❌ Input Error: Please enter numeric values for Amount, Hour, and Day.")
        except KeyboardInterrupt:
            print("\nSystem closed cleanly.")
            break