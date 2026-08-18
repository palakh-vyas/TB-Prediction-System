import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset

df = pd.read_csv("Dataset/Healthcare.csv")

# Create TB column
df["TB"] = df["Disease"].apply(lambda x: 1 if x == "Tuberculosis" else 0)

# Features and target
X = df["Symptoms"]
y = df["TB"]

# Convert text into vectors
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("Tuberculosis Risk Prediction System")

st.write("Enter patient symptoms below:")

# User input
user_input = st.text_input("Symptoms")

# Predict button
if st.button("Predict"):

    input_vector = vectorizer.transform([user_input])

    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        st.error("High Risk of Tuberculosis")
    else:
        st.success("Low Risk of Tuberculosis")
