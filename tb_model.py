import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(".vscode/Tuberculosis/dataset/Healthcare.csv")

# Create TB target column
df["TB"] = df["Disease"].apply(lambda x: 1 if x == "Tuberculosis" else 0)

# Features and target
X = df["Symptoms"]
y = df["TB"]

# Convert symptoms text into numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(accuracy)


# User input
user_input = input("\nEnter symptoms: ")

# Convert input into vector
input_vector = vectorizer.transform([user_input])

# Predict
prediction = model.predict(input_vector)

# Output
if prediction[0] == 1:
    print("\nHigh Risk of Tuberculosis")
else:
    print("\nLow Risk of Tuberculosis")

    