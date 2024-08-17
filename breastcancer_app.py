import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("data.csv")

# Drop columns that are not needed, including those with only missing values
data = data.drop(columns=['Unnamed: 32'], errors='ignore')

# Drop rows with missing values
data = data.dropna()

# Encode categorical 'diagnosis' column: 'M' -> 1, 'B' -> 0
label_encoder = LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

# Prepare features and target variable
X = data.drop("diagnosis", axis=1)
y = data['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# SHAP explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title("SHAP Analysis for Breast Cancer Classification")

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")
st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Summary plot
st.subheader("Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Input fields for features
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f"Enter {feature}:", value=float(X_test[feature].mean()), step=0.01)

# Create a DataFrame from input data
input_df = pd.DataFrame(input_data, index=[0])

# Make prediction
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]  # Probability of being malignant

# Display prediction
st.write(f"**Prediction:** {'Malignant' if prediction == 1 else 'Benign'}")
st.write(f"**Malignant Probability:** {probability:.2f}")

# SHAP explanation for the input
shap_values_input = explainer.shap_values(input_df)

# Force plot
st.subheader("Force Plot")
try:
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values_input[1], input_df))
except IndexError:
    st.write("Mismatch in SHAP values shape for force plot.")

# Decision plot
st.subheader("Decision Plot")
try:
    st_shap(shap.decision_plot(explainer.expected_value[1], shap_values_input[1], X.columns))
except IndexError:
    st.write("Mismatch in SHAP values shape for decision plot.")
