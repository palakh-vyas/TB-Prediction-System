import pandas as pd

# Load dataset
df = pd.read_csv("dataset/Healthcare.csv")

# Show first 5 rows
print(df.head())

# Show column names
print(df.columns)