import tensorflow as tf
import joblib
import pandas as pd
import numpy as np
import os
import argparse # Import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tensorflow warnings

def load_model_and_scaler(model_filepath, scaler_filepath, feature_cols_filepath):
    """Loads the trained model, scaler, and feature column names."""
    try:
        model = tf.keras.models.load_model(model_filepath)
        scaler = joblib.load(scaler_filepath)
        feature_columns = joblib.load(feature_cols_filepath)
        print("Model, scaler, and feature columns loaded successfully.")
        return model, scaler, feature_columns
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None, None, None

def preprocess_new_data(data, scaler, feature_columns):
    """Preprocesses new raw data using the loaded scaler and feature columns."""
    df = pd.DataFrame([data]) # Convert single data point to DataFrame

    # Ensure the new data has the same columns as the training data, fill missing with NaN
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Select and reorder columns to match the training data
    df = df[feature_columns]

    # Handle missing values (using the same strategy as in preprocessor.py, e.g., mean imputation)
    # In a real application, you should load saved imputation values (like means) from training
    # and use them here. For simplicity, we'll fill with 0 or a placeholder.
    # A more robust approach would involve saving and loading imputation values.
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
         df[col] = df[col].fillna(0) # Filling with 0 as a simple placeholder

    # Scale the features
    # Ensure the data is in the correct format (numpy array) for scaling
    X_scaled = scaler.transform(df.values)

    return X_scaled

def predict_mental_health_risk(model, preprocessed_data):
    """Predicts mental health risk using the loaded model."""
    prediction = model.predict(preprocessed_data)
    # Assuming binary classification with sigmoid output
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class[0], prediction[0][0] # Return class and probability

if __name__ == "__main__":
    model_path = 'E:/Deep learning/ANN/Mental_Health_Predictor/models/final_model.h5'
    scaler_path = 'E:/Deep learning/ANN/Mental_Health_Predictor/models/scaler.pkl'
    feature_cols_path = 'E:/Deep learning/ANN/Mental_Health_Predictor/models/feature_columns.pkl'

    model, scaler, feature_columns = load_model_and_scaler(model_path, scaler_path, feature_cols_path)

    if model and scaler and feature_columns is not None:
        print(f"Expected features: {feature_columns}")

        # Set up argument parser based on expected features
        parser = argparse.ArgumentParser(description='Predict mental health risk based on input features.')
        for feature in feature_columns:
            parser.add_argument(f'--{feature}', type=float, required=True, help=f'Value for feature: {feature}')

        args = parser.parse_args()

        # Create data dictionary from parsed arguments
        new_user_data = {}
        for feature in feature_columns:
            new_user_data[feature] = getattr(args, feature)

        print(f"Input data for prediction: {new_user_data}")

        # Preprocess the new data
        preprocessed_new_data = preprocess_new_data(new_user_data, scaler, feature_columns)

        # Make prediction
        predicted_class, probability = predict_mental_health_risk(model, preprocessed_new_data)

        print(f"\nPrediction for the new data point:")
        print(f"Predicted Mental Health Risk Class: {predicted_class}")
        print(f"Predicted Probability of Risk (Class 1): {probability:.4f}")

    else:
        print("Failed to load model, scaler, or feature columns. Cannot make predictions.")
