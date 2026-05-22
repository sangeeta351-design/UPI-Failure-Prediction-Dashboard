import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# 1. PAGE CONFIGURATION & STYLING
st.set_page_config(
    page_title="UPI Failure Predictor",
    page_icon="⚡",
    layout="wide"
)

# Custom Theme Adjustments via CSS Injection
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { width: 100%; background-color: #005088; color: white; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { background-color: #11caa0; color: white; }
    div[data-testid="stMetricValue"] { color: #005088; font-family: 'Merriweather', serif; }
    .success-box { padding: 20px; background-color: #d1fae5; border-left: 6px solid #10b981; border-radius: 8px; color: #065f46; }
    .fail-box { padding: 20px; background-color: #fee2e2; border-left: 6px solid #ef4444; border-radius: 8px; color: #991b1b; }
    </style>
""", unsafe_allow_html=True)


# 2. CACHED DATA PIPELINE & MODEL TRAINING
@st.cache_resource
def load_and_train_pipeline():
    # Load dataset
    df = pd.read_csv('transactions_upi_transaction_failure.csv')

    # Feature Engineering
    df['Bank'] = df['Sender UPI ID'].str.split('@').str[1]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek

    # Isolate targets and features
    X = df[['Amount (INR)', 'Bank', 'Hour', 'Day_of_Week']].copy()
    y = df['Status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)

    # Label encoding categorical variables
    le = LabelEncoder()
    X['Bank'] = le.fit_transform(X['Bank'])

    # Train-test split & core initialization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Score calculation
    acc = model.score(X_test, y_test)

    return model, le, acc, df, importance_dataframe(model, ['Amount (INR)', 'Bank', 'Hour', 'Day_of_Week'])


def importance_dataframe(model, feature_names):
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance Score': model.feature_importances_
    }).sort_values(by='Importance Score', ascending=False)
    return df_imp


# Core data instantiation
model, le, accuracy, raw_df, imp_df = load_and_train_pipeline()

# 3. SIDEBAR NAVIGATION & CONTROLS
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
st.sidebar.title("System Control Room")
st.sidebar.markdown("---")
st.sidebar.markdown("### **🔧 Live Simulation Parameters**")

# Interactive input sliders/dropdowns on sidebar panel
input_amount = st.sidebar.number_input("1. Transaction Amount (INR)", min_value=1.0, max_value=50000.0, value=500.0,
                                       step=50.0)
input_bank = st.sidebar.selectbox("2. Gateway Bank Handle", options=list(le.classes_))
input_hour = st.sidebar.slider("3. Time of Day (24hr Clock)", min_value=0, max_value=23, value=12)
input_day = st.sidebar.selectbox("4. Scheduled Day of Week",
                                 options=[0, 1, 2, 3, 4, 5, 6],
                                 format_func=lambda x:
                                 ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 Adjust the values above and click the check button to evaluate structural transactional safety loops.")

# 4. MAIN INTERFACE LAYOUT
st.title("⚡ Rural UPI Transaction Failure Analytics System")
st.markdown(
    "An advanced predictive data science dashboard mapping mobile banking infrastructure vulnerabilities across remote Indian demographics.")
st.markdown("---")

# Row 1: High Level Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Datasets Evaluated", f"{len(raw_df)} Rows")
with col2:
    st.metric("Model Baseline Accuracy", f"{accuracy * 100:.1f}%")
with col3:
    st.metric("Active Node Coverage", f"{len(le.classes_)} Core Banks")

st.markdown("### **🎯 Transaction Predictive Verification**")

# Evaluation execution block
encoded_bank = le.transform([input_bank])[0]
custom_data = pd.DataFrame([{
    'Amount (INR)': input_amount,
    'Bank': encoded_bank,
    'Hour': input_hour,
    'Day_of_Week': input_day
}])

prediction = model.predict(custom_data)[0]
probabilities = model.predict_proba(custom_data)[0]
fail_prob = probabilities[0] * 100
success_prob = probabilities[1] * 100

# UI Output layout handling based on classification status
if prediction == 1:
    st.markdown(f"""
    <div class="success-box">
        <h4>🟢 TRANSACTION APPROVED VIA SAFE GATEWAY</h4>
        <p>The system expects a stable, functional clearing route. 
        <b>Confidence Metric: {success_prob:.1f}%</b> chance of immediate structural success.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="fail-box">
        <h4>🔴 WARNING: HIGH FAILURE PROBABILITY DETECTED</h4>
        <p>This transaction vector hits systemic server or timeout limitations typically present in remote sectors. 
        <b>Risk Metric: {fail_prob:.1f}%</b> probability of gateway timeout.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Row 2: Charts and Explainability Data
left_graph_col, right_graph_col = st.columns(2)

with left_graph_col:
    st.markdown("#### 📊 System Feature Importance Breakdown")
    fig_imp = px.bar(
        imp_df,
        x='Importance Score',
        y='Feature',
        orientation='h',
        color='Feature',
        color_discrete_sequence=px.colors.sequential.Viridis,
        template='plotly_white'
    )
    fig_imp.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

with right_graph_col:
    st.markdown("#### ⏳ Network Activity Footprint (Historical Log Logs)")
    fig_time = px.histogram(
        raw_df,
        x='Hour',
        color='Status',
        barmode='group',
        color_discrete_map={'SUCCESS': '#11caa0', 'FAILED': '#005088'},
        template='plotly_white'
    )
    fig_time.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20),
                           legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_time, use_container_width=True)
