# Cancer-_Prediction-
# Cancer Prediction System

This project is a Flask-based web application designed for cancer prediction. It allows users to upload a dataset, train multiple machine learning models (SVC, Random Forest, and XGBoost), and make predictions using the trained models.

## Features
- Upload a CSV dataset for model training
- Train multiple models (SVC, Random Forest, XGBoost) using GridSearchCV
- View model accuracies after training
- Make predictions using a trained model

## Dataset
The system expects a CSV file with a 'diagnosis' column (binary classification) and an 'id' column. The 'diagnosis' column is encoded, and the 'id' column is dropped during preprocessing. This dataset is used for cancer prediction.

## Requirements
- Python 3.x
- Flask
- pandas
- numpy
- scikit-learn
- xgboost

Install dependencies with:
```bash
pip install Flask pandas numpy scikit-learn xgboost
```

## Usage
1. Place your dataset CSV file in the project directory.
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5000/`.
4. Upload your dataset to train models and view accuracies.
5. Use the trained models to make predictions by selecting a model and entering test data.

## File Structure
- `app.py` - Main Flask application
- `models/` - Directory to store trained models
- `templates/` - HTML templates for the web interface
    - `index.html` - Home page

## Example
Upload a dataset, train models, and view accuracies. Then, select a model and enter test data to get predictions for cancer diagnosis.

## License
This project is for educational purposes.
