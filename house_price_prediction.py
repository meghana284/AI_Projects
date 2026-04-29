from sklearn.linear_model import LinearRegression
import numpy as np

# Example data: [size (sq ft), bedrooms]
X = np.array([
    [1000, 2],
    [1500, 3],
    [2000, 4],
    [2500, 4],
    [3000, 5]
])

# Prices
y = np.array([200000, 300000, 400000, 450000, 500000])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for new house
new_house = np.array([[1800, 3]])
predicted_price = model.predict(new_house)

print("Predicted Price:", predicted_price[0])
