# Load libraries
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import joblib
import xgboost as xgb
from sklearn.model_selection import cross_val_score

# Load the CSV file into a DataFrame
df = pd.read_csv('data/properties.csv')

# Define features to use
cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'state_building', 
            'epc', 'heating_type', 'equipped_kitchen']
num_features = ["cadastral_income","surface_land_sqm", "total_area_sqm", "latitude", "longitude", "garden_sqm", 
            "primary_energy_consumption_sqm", "construction_year", "nbr_frontages", "nbr_bedrooms", "terrace_sqm" ]
dummy_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace","fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]

# Split the data into features and target
X = df[num_features + dummy_features + cat_features]
y = df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Impute missing values using SimpleImputer on numerical features
imputer = SimpleImputer(strategy="mean")
X_train[num_features] = imputer.fit_transform(X_train[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])

# Apply one-hot encoding to categorical variables
X_encoded = pd.get_dummies(X, columns=[
    'property_type', 'subproperty_type', 'region', 'province', 'locality', 
    'state_building', 'epc', 'heating_type', 'equipped_kitchen'
])

# train the model 
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=0)
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')
xgb_regressor.fit(X_train_encoded, y_train)

# Predict on the train data
y_pred_train = xgb_regressor.predict(X_train_encoded)
y_pred_test = xgb_regressor.predict(X_test_encoded)

# Calculate R-squared for properties
r2_train = r2_score(y_train, y_pred_train)
print("R-squared:", r2_train)
r2_test = r2_score(y_test, y_pred_test)
print("R-squared:", r2_test)

# Perform cross-validation
scores = cross_val_score(xgb_regressor, X_train_encoded, y_train, cv=5, scoring='neg_mean_squared_error')
mse_scores = -scores
print("Mean squared error for each fold:", mse_scores)
print("Average cross-validated MSE:", mse_scores.mean())

# Visualize the actual vs. predicted prices for training set
plt.figure(figsize=(8, 6))
plt.scatter(y_train, y_pred_train, color='blue', label='Predictions')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--', label='Actuals')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Training Set)')
plt.legend()
plt.show()

# Visualize the actual vs. predicted prices for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Actuals')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Test Set)')
plt.legend()
plt.show()

# Save the trained XGBoost model
joblib.dump(xgb_regressor, 'trained_xgb_model.pkl')