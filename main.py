import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress tensorflow warnings

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model import MentalHealthModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib # Import joblib
import pandas as pd # Import pandas

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('E:/Deep learning/ANN/Mental_Health_Predictor/training_history.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('E:/Deep learning/ANN/Mental_Health_Predictor/confusion_matrix.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    data_loader = DataLoader()
    data = data_loader.merge_data()

    print(f"Shape of loaded data: {data.shape}") # Added print statement

    if data.empty:
        print("Error: No data loaded")
        return

    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    # Pass the full data to prepare_data to get the feature columns used
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data)

    # Save the fitted scaler and feature column names
    scaler_filepath = 'E:/Deep learning/ANN/Mental_Health_Predictor/models/scaler.pkl'
    joblib.dump(preprocessor.scaler, scaler_filepath) # Save the scaler
    print(f"Fitted scaler saved to {scaler_filepath}")

    feature_cols_filepath = 'E:/Deep learning/ANN/Mental_Health_Predictor/models/feature_columns.pkl'
    joblib.dump(preprocessor.feature_columns, feature_cols_filepath) # Save feature column names
    print(f"Feature column names saved to {feature_cols_filepath}")

    print(f"Shape of X_train: {X_train.shape}") # Added print statement
    print(f"Shape of X_test: {X_test.shape}") # Added print statement
    print(f"Shape of y_train: {y_train.shape}") # Added print statement
    print(f"Shape of y_test: {y_test.shape}") # Added print statement

    # Create and train model
    print("Training model...")
    model = MentalHealthModel(input_dim=X_train.shape[1])
    history = model.train(
        X_train, y_train,
        X_test, y_test,  # Using test set as validation set for simplicity
        epochs=100,
        batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy, auc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))
    
    # Plot training history and confusion matrix
    print("\nGenerating plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Save the model
    print("\nSaving model...")
    model.save_model('E:/Deep learning/ANN/Mental_Health_Predictor/models/final_model.h5')
    
    print("\nDone! Model has been trained and saved.")

if __name__ == "__main__":
    main()
