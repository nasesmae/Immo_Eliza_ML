# Property Price Prediction XGBoost Model Card

## Model Overview

This XGBoost model predicts property prices using various property characteristics. The model utilizes both numerical and categorical data from a dataset containing detailed information on properties.

## Model Details

- **Developed by:** Nasrin Esmaeilian
- **Type:** Regression
- **Algorithm:** XGBoost (eXtreme Gradient Boosting)
- **Objective:** To predict property prices (`price`).

### Features

The model incorporates 31 features divided into three main categories:

#### Numerical Features
- `cadastral_income`
- `surface_land_sqm`
- `total_area_sqm`
- `latitude`
- `longitude`
- `garden_sqm`
- `primary_energy_consumption_sqm`
- `construction_year`
- `nbr_frontages`
- `nbr_bedrooms`
- `terrace_sqm`

#### Dummy Features
- `fl_garden`
- `fl_furnished`
- `fl_open_fire`
- `fl_terrace`
- `fl_swimming_pool`
- `fl_floodzone`
- `fl_double_glazing`

#### Categorical Features
- `property_type`
- `subproperty_type`
- `region`
- `province`
- `locality`
- `zip_code`
- `state_building`
- `epc`
- `heating_type`
- `equipped_kitchen`

### Encoding
- One-hot encoding is applied to categorical variables.

### Missing Value Handling
- Numerical features with missing values are imputed using the mean of the column.

## Training Details

- **Data Split:** 75% of the data is used for training, and 25% is used for testing.
- **Evaluation Metrics:** The model is evaluated using the R-squared value and cross-validated mean squared error (MSE).
- **Cross-Validation:** 5-fold cross-validation is performed.

## Performance

The model's ability to explain the variance in property prices is indicated by the R-squared value. Its generalization capability is assessed using cross-validated MSE.

## Visualizations

Scatter plots are generated to compare actual vs. predicted prices for both the training and testing datasets, providing a visual assessment of model performance.

## Model Storage

The trained model is serialized and stored using Joblib for easy retrieval and prediction in future applications.

## Usage

This model is intended for use by real estate agencies, property valuation companies, and individuals for estimating property market prices. It acts as a decision support tool in the real estate market for buying, selling, and investment purposes.

## Bias and Fairness Risks

- **Geographical Bias:** Since the model includes location-based features such as `region`, `province`, and `locality`, there's a risk of geographical bias, potentially leading to inaccurate price predictions for areas not well-represented in the training data.
- **Historical Bias:** The model's reliance on historical data, such as `construction_year`, may inadvertently perpetuate past market biases, especially in areas undergoing rapid development or change.
- **Feature Representation:** The choice and representation of features may not fully capture the nuances of property value, such as the quality of finishes, neighborhood amenities, or local economic factors, potentially leading to under- or over-estimation of prices.

## Limitations

- **Dynamic Market Conditions:** The model may not accurately predict prices in rapidly changing market conditions, as it is trained on historical data.
- **Generalization:** While the model is designed to generalize across various regions and property types, its accuracy may decrease for properties with unique or uncommon features not well-represented in the training set.
- **Data Quality:** The performance and reliability of the model are heavily dependent on the quality and completeness of the input data. Missing or inaccurate data can significantly impact predictions.
- **Interpretability:** The complexity of the XGBoost algorithm may make it challenging to understand the specific contributions of all features to the predicted prices, potentially complicating efforts to identify and correct biases.
