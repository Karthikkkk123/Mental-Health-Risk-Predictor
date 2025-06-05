import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None # Initialize feature_columns attribute
        
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df = df.copy()
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                # If more than 50% values are missing, drop the column
                if df[col].isnull().mean() > 0.5:
                    df = df.drop(col, axis=1)
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                # If more than 50% values are missing, drop the column
                if df[col].isnull().mean() > 0.5:
                    df = df.drop(col, axis=1)
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                    df[col] = df[col].fillna(mode_val)
        
        return df
    
    def extract_features(self, df):
        """Extract relevant features for mental health prediction"""
        # Always include depression_score as a feature if available
        features = ['depression_score'] if 'depression_score' in df.columns else []
        
        # Get all numeric columns except user_id and target variable
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features 
                          if f not in ['user_id', 'mental_health_risk'] 
                          and f not in features]
        features.extend(numeric_features)
        
        # Validate features
        if not features:
            raise ValueError("No valid features found in the dataset")
        
        # Check for highly correlated features
        if len(features) > 1:
            corr_matrix = df[features].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            features = [f for f in features if f not in to_drop]
        
        return df[features]
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        # Handle constant features
        constant_mask = np.all(X_train == X_train[0, :], axis=0)
        if any(constant_mask):
            X_train = X_train[:, ~constant_mask]
            if X_test is not None:
                X_test = X_test[:, ~constant_mask]
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """Prepare data for training"""
        # Handle missing values
        df = self.handle_missing_values(df)

        # Extract features
        X = self.extract_features(df)

        # Store the feature column names
        self.feature_columns = X.columns.tolist()

        # Ensure target variable exists
        if 'mental_health_risk' not in df.columns:
            raise ValueError("Target variable 'mental_health_risk' not found in dataset")

        y = df['mental_health_risk']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                           test_size=test_size,
                                                           random_state=random_state,
                                                           stratify=y)

        # Convert pandas DataFrames to numpy arrays before scaling
        X_train_np = X_train.values
        X_test_np = X_test.values

        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train_np, X_test_np)

        return X_train_scaled, X_test_scaled, y_train, y_test
