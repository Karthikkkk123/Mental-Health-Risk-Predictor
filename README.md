# 🧠 Mental Health Risk Predictor Based on Lifestyle and Digital Behavior (ANN)

## Overview
This project predicts an individual's mental health risk (e.g., stress, anxiety, depression) using an Artificial Neural Network (ANN) trained on data from lifestyle, screen usage, sleep patterns, and survey responses. It is designed for research, educational, and prototyping purposes.

## Features
- Predicts mental health risk using survey, app usage, and sensing data
- End-to-end pipeline: data loading, preprocessing, training, evaluation, and prediction
- CLI-based prediction for new individuals
- Includes full dataset for reproducibility

## Project Structure
```
Mental_Health_Predictor/
│   main.py                # Main training and evaluation script
│   predict.py             # CLI tool for making predictions on new data
│   requirements.txt       # Python dependencies
│   README.md              # Project documentation
│
├───src/                  # Source code for data loading, preprocessing, and model
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── model.py
│
├───models/               # Saved models and scalers
│   ├── best_model.h5
│   ├── final_model.h5
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├───notebooks/            # Jupyter notebooks for exploration
│   └── mental_health_predictor.ipynb
│
├───dataset/              # Full dataset (app usage, survey, sensing, etc.)
│   ├── app_usage/
│   ├── survey/
│   └── ...
│
├───data/                 # (Optional) Processed or intermediate data
│
├───confusion_matrix.png  # Confusion matrix plot from last run
├───training_history.png  # Training/validation loss and accuracy plot
```

## How to Run
### 1. Install Dependencies
Make sure you have Python 3.8+ and pip installed. Then run:
```pwsh
pip install -r requirements.txt
```

### 2. Train the Model
This will load the dataset, preprocess, train the ANN, evaluate, and save the model and scaler:
```pwsh
python main.py
```
- Outputs: `models/final_model.h5`, `models/scaler.pkl`, `models/feature_columns.pkl`, plots, and evaluation metrics.

### 3. Predict Mental Health Risk for New Data
You can predict for a new individual using the CLI. The script will print the required features (e.g., `depression_score`). Example:
```pwsh
python predict.py --depression_score 12
```
- Replace `12` with the actual PHQ-9 score or other required features.
- The script will output the predicted risk class (0: low, 1: high) and probability.

### 4. Dataset
- The `dataset/` folder contains all raw data used for training and testing.
- Subfolders include: `app_usage/`, `survey/`, `sensing/`, etc.
- Example survey file: `dataset/survey/PHQ-9.csv` (PHQ-9 depression scores)

## Inputs Explained
- **depression_score**: Total score from the PHQ-9 survey (integer, 0-27). Higher values indicate higher risk of depression.
- If more features are required (e.g., activity stats, app usage), the script will print their names. Provide them as CLI arguments.

## What Happens When You Run
- `main.py` loads and merges all data, preprocesses it, trains an ANN, evaluates, saves the model, scaler, and feature list, and generates plots.
- `predict.py` loads the trained model and scaler, takes CLI input for features, preprocesses, and predicts the risk.

## Notes
- The dataset is large (~3GB). Make sure you have enough disk space and memory.
- For research/educational use only. Not for clinical or diagnostic use.
- You can explore and modify the code in the `src/` folder and the Jupyter notebook in `notebooks/`.

## License
This project is for educational and research purposes. Please check the dataset sources for their respective licenses.

---

**Author:** Karthikeya (Karthikkkk123)
