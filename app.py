from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import os

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = "secret_key_for_flash_messages"

# Ensure the /models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Preprocess data function
def preprocess_data(df):
    # Encoding 'diagnosis' column and dropping 'id' column
    df['diagnosis'] = df['diagnosis'].astype('category').cat.codes
    X = df.drop(columns=['diagnosis', 'id'])
    y = df['diagnosis']
    
    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function to fit and save model using GridSearchCV
def FitModel(X, y, model_name, algorithm, params, cv=10):
    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Perform grid search
    grid = GridSearchCV(estimator=algorithm, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    
    # Save the trained model
    model_path = f'./models/{model_name}.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(grid, file)
    
    # Return the accuracy on the test set
    accuracy = grid.score(x_test, y_test)
    return accuracy

# Route for model training
@app.route('/train', methods=['POST'])
def train_model():
    if 'file' not in request.files:
        flash('No file uploaded!')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file!')
        return redirect(url_for('index'))
    
    if file:
        # Load dataset
        df = pd.read_csv(file)
        
        # Preprocess dataset
        X, y = preprocess_data(df)
        
        # SVC hyperparameters
        svm_params = {
            'C': [0.1, 1, 100],
            'gamma': [0.0001, 0.001, 0.01]
        }
        
        # Train and save SVC model
        svc_accuracy = FitModel(X, y, 'SVC', SVC(), svm_params)
        
        # Random Forest hyperparameters
        rf_params = {'n_estimators': [100, 500]}
        
        # Train and save Random Forest model
        rf_accuracy = FitModel(X, y, 'RandomForest', RandomForestClassifier(), rf_params)

        # XGBoost hyperparameters
        xgb_params = {'n_estimators': [100, 500]}
        
        # Train and save XGBoost model
        xgb_accuracy = FitModel(X, y, 'XGBoost', XGBClassifier(), xgb_params)
        
        # Flash model accuracies
        flash(f'SVC Accuracy: {svc_accuracy * 100:.2f}%')
        flash(f'RandomForest Accuracy: {rf_accuracy * 100:.2f}%')
        flash(f'XGBoost Accuracy: {xgb_accuracy * 100:.2f}%')
        
        return redirect(url_for('index'))

# Route for loading a model and making predictions
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    model_path = f'./models/{model_name}.pkl'
    
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    # Get user input for prediction (assuming user input is passed as comma-separated values)
    input_data = request.form['test_data']
    try:
        # Convert input string to float values (assuming CSV format)
        test_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
    except ValueError:
        flash('Invalid input. Please enter numeric values.')
        return redirect(url_for('index'))

    # Make prediction
    prediction = model.predict(test_data)
    
    flash(f'Model {model_name} predicted: {prediction[0]}')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
