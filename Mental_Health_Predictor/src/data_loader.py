import pandas as pd
import numpy as np
from pathlib import Path
import os

class DataLoader:
    def __init__(self, base_path="E:/dataset"):
        self.base_path = Path(base_path)

    def _extract_user_id(self, filename):
        """Extract user ID from filename"""
        # Handle potential variations in filename format
        parts = filename.stem.split('_')
        return parts[1] if len(parts) > 1 else filename.stem

    def load_app_usage_data(self):
        """Load and process app usage data"""
        app_path = self.base_path / "app_usage"
        if not app_path.exists():
            print(f"Warning: App usage data path not found at {app_path}")
            return pd.DataFrame()

        app_dfs = []
        for file in app_path.glob("*.csv"):
            try:
                user_id = self._extract_user_id(file)
                df = pd.read_csv(file)
                if not df.empty:
                    df['user_id'] = user_id
                    app_dfs.append(df)
            except Exception as e:
                print(f"Warning: Error processing app usage file {file}: {str(e)}")
                continue

        print(f"Loaded {len(app_dfs)} app usage dataframes.") # Debug print
        return pd.concat(app_dfs) if app_dfs else pd.DataFrame()

    def load_sensing_data(self):
        """Load and process sensing data"""
        activity_path = self.base_path / "sensing" / "activity"
        if not activity_path.exists():
            print(f"Warning: Activity data path not found at {activity_path}")
            return pd.DataFrame()

        activity_dfs = []
        for file in activity_path.glob("activity_*.csv"):
            try:
                user_id = self._extract_user_id(file)

                # Read the CSV file with header
                df = pd.read_csv(file)

                # Find timestamp and activity columns robustly
                timestamp_col = None
                activity_col = None
                for col in df.columns:
                    lower_col = col.lower().strip()
                    if 'timestamp' in lower_col:
                        timestamp_col = col
                    if 'activity' in lower_col:
                        activity_col = col

                print(f"Processing file {file}, original columns: {df.columns.tolist()}") # Debug print
                print(f"Detected timestamp column: {timestamp_col}, activity column: {activity_col}") # Debug print

                if not timestamp_col or not activity_col:
                    print(f"Warning: Skipping file {file} due to missing required columns (timestamp or activity).")
                    continue

                # Rename columns to standard names
                df = df.rename(columns={timestamp_col: 'timestamp', activity_col: 'activity_level'})

                # Convert timestamp to numeric, then to datetime, coercing errors
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

                # Convert activity_level to numeric, coercing errors
                df['activity_level'] = pd.to_numeric(df['activity_level'], errors='coerce')

                # Drop rows with missing values in essential columns
                df.dropna(subset=['timestamp', 'activity_level'], inplace=True)

                print(f"File {file} after cleaning: Shape {df.shape}, Dtypes {df.dtypes.to_dict()}") # Debug print

                if not df.empty and pd.api.types.is_numeric_dtype(df['activity_level']):
                    # Calculate daily activity statistics
                    daily_stats = df.groupby(['user_id', df['timestamp'].dt.date])['activity_level'].agg([
                        'mean', 'std', 'max', 'min', 'count'
                    ]).reset_index()

                    daily_stats.columns = ['user_id', 'date',
                                         'daily_activity_mean',
                                         'daily_activity_std',
                                         'daily_activity_max',
                                         'daily_activity_min',
                                         'daily_activity_count']

                    activity_dfs.append(daily_stats)
                    print(f"Successfully processed and aggregated data for file {file}.") # Debug print
                elif not df.empty:
                     print(f"Warning: Skipping aggregation for file {file} as activity_level is not numeric after cleaning or dataframe is empty. Dtype: {df['activity_level'].dtype}") # Debug print
                else:
                    print(f"Warning: Skipping aggregation for file {file} as dataframe is empty after cleaning.") # Debug print

            except Exception as e:
                print(f"Warning: Error processing file {file}: {str(e)}")
                continue

        if not activity_dfs:
            print("Warning: No valid activity dataframes were processed.")
            return pd.DataFrame()

        # Combine all users' data and calculate average stats per user
        all_activity = pd.concat(activity_dfs)
        user_stats = all_activity.groupby('user_id').agg({
            'daily_activity_mean': 'mean',
            'daily_activity_std': 'mean',
            'daily_activity_max': 'max',
            'daily_activity_min': 'min',
            'daily_activity_count': 'mean'
        }).reset_index()

        print(f"Shape of processed activity data: {user_stats.shape}") # Debug print
        return user_stats

    def load_phq9_data(self):
        """Load and process PHQ-9 survey data"""
        phq9_path = self.base_path / "survey" / "PHQ-9.csv"
        if not phq9_path.exists():
            raise FileNotFoundError("PHQ-9.csv not found in survey directory")

        df = pd.read_csv(phq9_path)

        # Rename uid to user_id if present
        if 'uid' in df.columns:
            df = df.rename(columns={'uid': 'user_id'})

        # Map responses to scores
        response_map = {
            'Not at all': 0,
            'Several days': 1,
            'More than half the days': 2,
            'Nearly every day': 3
        }

        # Get question columns (excluding metadata columns)
        question_cols = [col for col in df.columns 
                        if col not in ['user_id', 'uid', 'type', 'Response']]

        # Convert responses to scores
        for col in question_cols:
            df[col] = df[col].map(response_map)

        # Calculate total score and risk label
        df['depression_score'] = df[question_cols].sum(axis=1)
        df['mental_health_risk'] = (df['depression_score'] >= 10).astype(int)

        # Keep only necessary columns
        return df[['user_id', 'depression_score', 'mental_health_risk']]

    def merge_data(self):
        """Merge all data sources and create feature matrix"""
        # Load PHQ-9 data (our target variable)
        phq9_data = self.load_phq9_data()
        print(f"Shape of PHQ-9 data: {phq9_data.shape}") # Debug print

        # Load feature data
        activity_data = self.load_sensing_data()
        app_data = self.load_app_usage_data()

        # Start with PHQ-9 data
        merged_data = phq9_data

        # Merge activity data if available and not empty
        if not activity_data.empty:
            merged_data = pd.merge(merged_data, activity_data, on='user_id', how='left')
            print(f"Shape after merging activity data: {merged_data.shape}") # Debug print
        else:
            print("Warning: Activity data is empty or failed to load, skipping merge.")

        # Merge app usage data if available and not empty
        if not app_data.empty:
            merged_data = pd.merge(merged_data, app_data, on='user_id', how='left')
            print(f"Shape after merging app usage data: {merged_data.shape}") # Debug print
        else:
            print("Warning: App usage data is empty or failed to load, skipping merge.")

        print(f"Shape of final merged data: {merged_data.shape}") # Debug print
        print(f"Columns of final merged data: {merged_data.columns.tolist()}") # Debug print
        print(f"Data types of final merged data:\n{merged_data.dtypes}") # Debug print

        return merged_data
