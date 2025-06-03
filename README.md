# Mental Health Risk Predictor

An AI-powered Mental Health Risk Assessment Tool that uses Artificial Neural Networks to predict potential mental health risks based on sleep patterns, alcohol consumption, and medical history. The model is trained on the NHANES (National Health and Nutrition Examination Survey) dataset.

## Features

- **Neural Network Model**: Built with TensorFlow/Keras for accurate risk prediction
- **Interactive Assessment**: User-friendly questionnaire covering key health aspects
- **Comprehensive Analysis**: Evaluates multiple risk factors:
  - Sleep patterns and quality
  - Alcohol consumption habits
  - Medical history and conditions
- **Privacy-Focused**: All processing is done locally
- **Professional Recommendations**: Provides appropriate guidance based on risk levels

## Model Performance

The Neural Network model achieves robust performance in risk prediction:
- High accuracy in identifying potential mental health risks
- ROC-AUC score demonstrates strong discriminative ability
- Balanced precision and recall for both high and low-risk cases

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karthikkkk123/mental-health-risk-predictor.git
cd mental-health-risk-predictor
```

2. Create and activate a Python 3.10 virtual environment:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start Jupyter Notebook:
```bash
jupyter notebook mental_health_predictor.ipynb
```

2. Run all cells in the notebook to:
   - Load and process the NHANES dataset
   - Train the neural network model
   - Access the interactive risk assessment tool

3. Follow the interactive prompts to:
   - Answer questions about sleep patterns
   - Provide information about alcohol consumption
   - Share relevant medical history
   - Receive a risk assessment and recommendations

## Data Processing Pipeline

1. **Data Loading**: Imports and merges relevant NHANES dataset components
2. **Cleaning**: Handles missing values and removes anomalies
3. **Feature Engineering**: Creates composite risk scores from multiple factors
4. **Normalization**: Scales numerical features for optimal model performance
5. **Encoding**: Converts categorical variables into model-compatible format

## Model Architecture

- Input Layer: Matches preprocessed feature dimensions
- Hidden Layers:
  - Dense layer (64 units) with ReLU activation
  - Dropout (0.3) for regularization
  - Dense layer (32 units) with ReLU activation
  - Dropout (0.2)
  - Dense layer (16 units) with ReLU activation
- Output Layer: Single unit with sigmoid activation for binary classification

## Important Notes

- This tool is for screening purposes only and not a substitute for professional medical advice
- Always consult healthcare professionals for proper diagnosis and treatment
- Keep your responses honest and accurate for better assessment results
- Regular reassessment is recommended as mental health status can change

## Requirements

- Python 3.10
- TensorFlow 2.10.0
- NumPy 1.23.5
- Pandas 1.5.3
- Matplotlib 3.7.1
- Seaborn 0.12.2
- Scikit-learn 1.2.2

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue in the GitHub repository.