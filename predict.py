import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os

def preprocess_data(data):
    # Encode non-numeric columns
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data, label_encoders

def train_classification_model(file_name, model_name, scaler_name):
    try:
        # Load dataset
        data = pd.read_csv(file_name)
        print(f"Loaded {file_name} successfully.")

        # Preprocess data
        data, label_encoders = preprocess_data(data)

        # Separate features and target variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{file_name} Model Accuracy: {accuracy:.2f}")

        # Save the scaler and model
        print(f"Saving scaler to {scaler_name}")
        joblib.dump(scaler, scaler_name)
        print(f"Scaler saved to {scaler_name}")

        print(f"Saving model to {model_name}")
        joblib.dump(model, model_name)
        print(f"Model saved to {model_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

def train_regression_model(file_name, model_name, scaler_name):
    try:
        # Load dataset
        data = pd.read_csv(file_name)
        print(f"Loaded {file_name} successfully.")

        # Preprocess data
        data, label_encoders = preprocess_data(data)

        # Separate features and target variable
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{file_name} Model Mean Squared Error: {mse:.2f}")

        # Save the scaler and model
        print(f"Saving scaler to {scaler_name}")
        joblib.dump(scaler, scaler_name)
        print(f"Scaler saved to {scaler_name}")

        print(f"Saving model to {model_name}")
        joblib.dump(model, model_name)
        print(f"Model saved to {model_name}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Train models for each dataset
train_classification_model('diabetes.csv', 'model1.pkl', 'scaler1.pkl')
train_classification_model('heart.csv', 'model2.pkl', 'scaler2.pkl')
train_regression_model('parkinsons.csv', 'model3.pkl', 'scaler3.pkl')  # Use regression model for Parkinson's dataset
