
# AI CyberShield

AI CyberShield is a machine learning-based intrusion detection system that classifies network traffic into different types of attacks or normal activity using trained classification models.

## Features

- Preprocessing of raw network traffic datasets
- Feature selection and encoding
- Training multiple machine learning models
- Evaluation and selection of best model
- Real-time intrusion prediction using saved model

## Project Structure

```
.
├── ai_model.py                 # Main model logic
├── create_dataset.py          # Script to generate final dataset
├── intrusion_detection.py     # CLI to run prediction on new input
├── test_prediction.py         # Script to test final prediction output
├── evaluate_model.py          # Model evaluation pipeline
├── data_analysis.py           # Exploratory data analysis
├── utils/                     # Preprocessing and feature selection code
├── model/                     # Contains all saved model and scaler files
├── requirements.txt           # Dependencies list
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pawan872633/AI_CyberShield.git
   cd AI_CyberShield
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/Scripts/activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use

### 1. Create Dataset
Run the script to generate a clean dataset from raw data.
```bash
python create_dataset.py
```

### 2. Train Model
Use the training script to train and save the model.
```bash
python ai_model.py
```

### 3. Make a Prediction
Run the CLI tool to test a new prediction:
```bash
python test_prediction.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

- **Pawan Kumar** – [GitHub Profile](https://github.com/pawan872633)
