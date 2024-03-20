import pandas as pd
import numpy as np
import os
import joblib


# Function to load the trained model and pre-fitted transformers
def load_model_and_transformers(model_path, imputer_path, encoder_path):
    try:
        regressor = joblib.load(model_path)
        imputer = joblib.load(imputer_path)
        encoder = joblib.load(encoder_path)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        raise
    return regressor, imputer, encoder

# Function to preprocess new data
def preprocess_data(new_data, imputer, encoder, num_features, dummy_features, cat_features):
    try:
        # Ensure new_data has the same column order as during fitting
        # Sort the columns of new_data to match the training data order
        new_data = new_data[num_features + dummy_features + cat_features]
        
        # Impute missing values for numerical features
        new_data_num = imputer.transform(new_data[num_features])
        
        # Encode categorical features
        encoded_features = encoder.transform(new_data[cat_features]).toarray()
        
        # Concatenate numerical, boolean, and encoded categorical features
        X = np.concatenate([new_data_num, new_data[dummy_features].values, encoded_features], axis=1)
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        raise
    return X


def main():
    model_path = 'trained_model_3.pkl'
    imputer_path = 'imputer.pkl'
    encoder_path = 'encoder.pkl'

    # Define numerical, dummy, and categorical features
    cat_features = ['property_type', 'subproperty_type', 'region', 'province', 'locality', 'zip_code', 'state_building', 
            'epc', 'heating_type', 'equipped_kitchen']
    num_features = ["cadastral_income","surface_land_sqm", "total_area_sqm", "latitude", "longitude", "garden_sqm", 
            "primary_energy_consumption_sqm", "construction_year", "nbr_frontages", "nbr_bedrooms", "terrace_sqm" ]
    dummy_features = ["fl_garden", "fl_furnished", "fl_open_fire", "fl_terrace","fl_swimming_pool", "fl_floodzone", "fl_double_glazing"]


    # Load the trained model and pre-fitted transformers
    regressor, imputer, encoder = load_model_and_transformers(model_path, imputer_path, encoder_path)

    # Load new data
    file_path = input("Enter the path to the new data file: ").strip()  # Adjust for cross-platform compatibility
    try:
        new_data = pd.read_csv(file_path)  # Use the variable for dynamic path
    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        return
    
    # Preprocess the data
    X = preprocess_data(new_data, imputer, encoder, num_features, dummy_features, cat_features)

    # Make predictions
    predictions = regressor.predict(X)

    # Save predictions to a CSV file
    output_file = 'property_price_predictions.csv'
    pd.DataFrame(predictions, columns=['Predicted_Price']).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    main()

