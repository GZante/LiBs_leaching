import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Path to the database containing leaching values of one single metal for best results
# file_path = "path_to_your_database.xlsx"

# Load data from Excel file
df = pd.read_excel(file_path)

# Filter D values
df_filtered = df[(df['E'] >= 0.1) & (df['E'] <= 0.95)]

# Preprocessing
features = df_filtered.drop(columns=["E"])
target = df_filtered["E"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize XGBRegressor
xgb = XGBRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best model and its hyperparameters
best_model = grid_search.best_estimator_
best_hyperparams = grid_search.best_params_

# Evaluate the best model on training set
y_train_pred = best_model.predict(X_train)

# Metrics on training set
r_squared_train = r2_score(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
aard_train = (abs((y_train - y_train_pred) / y_train)).mean() * 100

# Evaluate the best model on test set
y_test_pred = best_model.predict(X_test)

# Metrics on test set
r_squared_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
aard_test = (abs((y_test - y_test_pred) / y_test)).mean() * 100

# Results
print("Training Metrics:")
print(f"R2: {r_squared_train}, MAE: {mae_train}, RMSE: {rmse_train}, AARD: {aard_train}")
print("Testing Metrics:")
print(f"R2: {r_squared_test}, MAE: {mae_test}, RMSE: {rmse_test}, AARD: {aard_test}")
print(f"Best Model: {best_model}")
print(f"Best Hyperparameters: {best_hyperparams}")

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue', label='Experimental vs Calculated E')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Best linear fit')
plt.xlabel('Experimental E')
plt.ylabel('Calculated E')
plt.legend(frameon=False)
plt.show()
