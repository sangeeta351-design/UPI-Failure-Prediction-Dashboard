# ⚡ Predictive Analytics System for Rural UPI Transaction Failures

An end-to-end data science pipeline and interactive web application designed to predict and diagnose mobile banking infrastructure vulnerabilities across remote demographics. Built entirely in Python using machine learning frameworks and deployed as a real-time analytics dashboard.

---

## 🎯 Project Overview & Objective
In urban ecosystems, digital payment failures are typically attributed to basic user issues like incorrect PIN entries or insufficient funds. However, in **rural sectors**, transactional drops are heavily driven by systematic infrastructure bottlenecks:
* High network latency and packet loss across fluctuating 2G/3G/4G carrier bands.
* High vulnerability to legacy banking node timeouts during localized peak traffic windows.
* Unannounced bank central server maintenance cycles scheduled during evening rural business windows.

**The Solution:** This project utilizes an ensemble machine learning model to evaluate core transactional parameters—such as transaction sizes, specific gateway handlers, and temporal attributes—to predict failure risks before the banking instruction is routed. This allows application logic to warn users or dynamically select safer routing nodes.

---

## 📊 Project Interface & Visual Dashboard
Our machine learning pipeline is fully operational via an interactive user dashboard built using Streamlit and Plotly vector visuals:

---

## 🛠️ System Architecture & File Directory
The repository is modularly organized into distinct sequential phases mirroring the standard Data Science Lifecycle:

```text
📂 PythonProject4/
├── 📄 transactions_upi_transaction_failure.csv  # 1,000-row historical transaction dataset
├── 📄 eda.py                                    # Exploratory data distribution analysis
├── 📄 preprocessing.py                          # Data cleaning, matrix structuring, and label encoding
├── 📄 final_model.py                            # Core model training, validation, and feature analysis
├── 📄 predict.py                                # Dedicated terminal-loop execution engine for reviewers
└── 📄 app.py                                    # Production-grade user dashboard interface
