# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 2: Create the Dataset using NumPy Arrays
body_weight = np.array([40, 45, 50, 55, 60, 65, 70])
protein_intake = np.array([72, 81, 90, 99, 108, 117, 126])

# Reshaping X is crucial for sklearn (converts it from a list [1,2] to a column [[1],[2]])
X = body_weight.reshape(-1, 1)
y = protein_intake

# Step 3: Explore the Data
print("--- Data showcase ---")
print(f"Body Weight (X):\n{X}")
print(f"Protein Intake (y):\n{y}")

# Step 4: Visualize the Data
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='green', s=100, label='Known Data')
plt.title('Body Weight vs Protein Intake', fontsize=14)
plt.xlabel('Body Weight (kg)', fontsize=12)
plt.ylabel('Protein Intake (g)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Step 5: Train the Model
model = LinearRegression()
model.fit(X, y)

print(f"\nModel Trained successfully.")
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Step 6: Visualize the Regression Line
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='green', s=100, label='Actual Data')

# Plotting the prediction line
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')

plt.title('Linear Regression Fit', fontsize=14)
plt.xlabel('Body Weight (kg)')
plt.ylabel('Protein Intake (g)')
plt.legend()
plt.show()

# Step 7: Predict for New Data
new_weight = 57
# We must double brackets [[ ]] to make it a 2D array for the prediction
predicted_protein = model.predict([[new_weight]])

print(f"\n--- Prediction Result ---")
print(f"For a person weighing {new_weight} kg,Recommended Protein Intake is : {predicted_protein[0]:.2f} grams")
