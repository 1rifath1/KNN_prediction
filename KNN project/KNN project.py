

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

# Load the dataset
data = load_breast_cancer()

# Create a DataFrame from the data
df = pd.DataFrame(np.c_[data.data, data.target], columns=list(data.feature_names) + ['target'])

# Split the data into training and test sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

# Train the KNN classifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

# Streamlit app
st.title("Breast Cancer Prediction App")

st.write("""
This app predicts whether a breast cancer case is malignant or benign based on input features.
""")

# Input fields for all features
input_data = []
for feature in data.feature_names:
    value = st.number_input(f"Input {feature}", 0.0)
    input_data.append(value)

# Convert input data to a DataFrame
input_data = np.array(input_data).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = neigh.predict(input_data)
    prediction_proba = neigh.predict_proba(input_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("The prediction is: **Malignant**")
    else:
        st.write("The prediction is: **Benign**")

    # Display prediction probabilities
    st.write(f"Probability of being benign: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of being malignant: {prediction_proba[0][1]:.2f}")

# Display model accuracy
st.write(f"Model Accuracy: {neigh.score(X_test, y_test):.2f}")
