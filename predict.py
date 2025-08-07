import joblib
import pandas as pd
import os
import numpy as np

MODEL_PATHS = {
    "Diabetes": "model1.pkl",
    "Heart Disease": "model2.pkl",
    "Parkinson's": "model3.pkl"
}

HEART_COMPONENTS = {
    "scaler": "scaler2.pkl",
    "selector": "features_model2.pkl"
}

# Complete feature mapping (UI names → CSV names)
FEATURE_MAPPING = {
    "Heart Disease": {
        "Age": "age",
        "Sex": "sex",
        "ChestPainType": "cp",
        "RestingBP": "trestbps",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "RestingECG": "restecg",
        "MaxHR": "thalach",
        "ExerciseAngina": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
        "Ca": "ca",
        "Thal": "thal"
    }
}

def predict_disease(disease, user_input):
    """Predict disease using the appropriate model"""
    try:
        # Default response for empty inputs
        if all(v == 0 for v in user_input.values()):
            return f"No {disease} Detected (Default - No data entered)"
        
        # Verify model exists
        if not os.path.exists(MODEL_PATHS[disease]):
            return f"Error: Model file not found for {disease}"
        
        model = joblib.load(MODEL_PATHS[disease])
        
        # Special handling for Heart Disease
        if disease == "Heart Disease":
            # Load preprocessing components
            scaler = joblib.load(HEART_COMPONENTS["scaler"])
            selector = joblib.load(HEART_COMPONENTS["selector"])
            
            # Map all 13 features to CSV column names
            mapped_input = {}
            for app_name, model_name in FEATURE_MAPPING[disease].items():
                mapped_input[model_name] = user_input[app_name]
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([mapped_input.values()], 
                                  columns=mapped_input.keys())
            
            # Apply preprocessing
            X_scaled = scaler.transform(input_df)
            X_selected = selector.transform(X_scaled)
            
            # Predict
            prediction = model.predict(X_selected)[0]
        else:
            # Original logic for other diseases
            if disease in FEATURE_MAPPING:
                mapped_input = {}
                for app_name, model_name in FEATURE_MAPPING[disease].items():
                    mapped_input[model_name] = user_input[app_name]
                # Add unmapped features
                for feature in user_input:
                    if feature not in mapped_input:
                        mapped_input[feature] = user_input[feature]
                user_input = mapped_input
            
            input_df = pd.DataFrame([user_input.values()], columns=user_input.keys())
            prediction = model.predict(input_df)[0]
        
        return f"{disease} Detected" if prediction == 1 else f"No {disease} Detected"
    
    except Exception as e:
        return f"Error: {str(e)}"

def load_average_values(disease):
    """Load and calculate average values from dataset"""
    CSV_PATHS = {
        "Diabetes": "diabetes.csv",
        "Heart Disease": "heart.csv",
        "Parkinson's": "parkinsons.csv"
    }
    
    try:
        if disease not in CSV_PATHS or not os.path.exists(CSV_PATHS[disease]):
            return {}
            
        df = pd.read_csv(CSV_PATHS[disease])
        
        # Convert categorical columns to numeric
        categorical_cols = ['cp', 'restecg', 'slope', 'thal'] if disease == "Heart Disease" else []
        for col in categorical_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop target column if exists
        target_cols = [col for col in df.columns if col.lower() in ["target", "status"]]
        if target_cols:
            df = df.drop(columns=target_cols)
        
        # Calculate averages and handle missing values
        averages = df.mean(numeric_only=True).fillna(0).to_dict()
        
        # Map CSV column names back to display names for Heart Disease
        if disease == "Heart Disease":
            display_mapping = {v: k for k, v in FEATURE_MAPPING[disease].items()}
            averages = {display_mapping.get(k, k): v for k, v in averages.items()}
        
        return averages
        
    except Exception as e:
        print(f"Error loading averages: {e}")
        return {}
