import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("house_data.csv")

# Features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
area = int(input("Enter house area (sq.ft): "))
bedrooms = int(input("Enter number of bedrooms: "))

# Predict
price = model.predict([[area, bedrooms]])

print("\n========== RESULT ==========")
print("Area:", area)
print("Bedrooms:", bedrooms)
print("Predicted House Price: ₹", round(price[0], 2))

# Graph
plt.scatter(df["Area"], df["Price"])
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price (₹)")
plt.title("House Price Prediction")
plt.show()