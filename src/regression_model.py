
# ==================================================
# CNC SURFACE ROUGHNESS PREDICTION
# ALGORITHM: LINEAR REGRESSION
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score

# STEP 1: Locate project folder
project_folder = Path(__file__).resolve().parent.parent

# STEP 2: Load cleaned dataset
file_path = project_folder / "outputs" / "cleaned_cnc_data.csv"
data = pd.read_csv(file_path)

# STEP 3: Select input and output variables
features = ["fz", "ap", "Vc", "vb", "C"]

X = data[features]
y = data["Ra"]

# STEP 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# STEP 5: Create Linear Regression model
model = LinearRegression()

# STEP 6: Train the model
model.fit(X_train, y_train)

# STEP 7: Make predictions
y_pred = model.predict(X_test)

# STEP 8: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nLINEAR REGRESSION RESULTS")
print("-------------------------")
print("MAE:", round(mae, 4))
print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# STEP 9: Display model coefficients
print("\nMODEL COEFFICIENTS")
print("Intercept:", model.intercept_)

for feature, coefficient in zip(features, model.coef_):
    print(feature, ":", round(coefficient, 4))

# STEP 10: Compare actual and predicted values
results = pd.DataFrame({
    "Actual Ra": y_test.values,
    "Predicted Ra": y_pred
})

print("\nACTUAL VS PREDICTED")
print(results.round(4))

# STEP 11: Plot actual vs predicted values
plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Surface Roughness (Ra)")
plt.ylabel("Predicted Surface Roughness (Ra)")
plt.title("Linear Regression: Actual vs Predicted")

plt.savefig(
    project_folder / "outputs" / "regression_results.png"
)

plt.show()

print("\nRegression model completed!")
