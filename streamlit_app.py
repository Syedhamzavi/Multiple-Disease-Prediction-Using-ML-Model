import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load models and scalers
def load_model_and_scaler(model_paths, scaler_paths):
    models = {}
    scalers = {}
    for disease, (model_path, scaler_path) in model_paths.items():
        try:
            models[disease] = joblib.load(model_path)
            scalers[disease] = joblib.load(scaler_path)
        except Exception as e:
            st.error(f"Error loading {disease} model or scaler: {e}")
            models[disease] = None
            scalers[disease] = None
    return models, scalers

# Function to load datasets
def load_datasets(file_paths):
    datasets = {}
    for disease, file_path in file_paths.items():
        try:
            datasets[disease] = pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading {disease} data: {e}")
            datasets[disease] = None
    return datasets

# Load models, scalers, and datasets
model_paths = {
    'DIABETES': ('model1.pkl', 'scaler1.pkl'),
    'HEART': ('model2.pkl', 'scaler2.pkl'),
    'PARKINSON': ('model3.pkl', 'scaler3.pkl')
}
file_paths = {
    'DIABETES': 'diabetes.csv',
    'HEART': 'heart.csv',
    'PARKINSON': 'parkinsons.csv'
}
models, scalers = load_model_and_scaler(model_paths, scaler_paths=model_paths)
datasets = load_datasets(file_paths)

# Define the Streamlit app
st.title('Disease Prediction App')

# Sidebar for selecting the disease
disease = st.sidebar.selectbox('Select Disease', list(file_paths.keys()))

# Input fields for the selected disease
with st.form(key='input_form'):
    st.header(f'{disease} Prediction')
    feature_names = datasets[disease].columns[:-1]  # Exclude target variable
    input_data = [st.number_input(feature, step=0.01) for feature in feature_names]
    submit_button = st.form_submit_button(label='Predict')

# Prediction logic
if submit_button:
    input_data = np.array([input_data])
    model = models.get(disease)
    scaler = scalers.get(disease)
    
    if model and scaler:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        probability = model.predict_proba(input_data_scaled)[0][1] if hasattr(model, 'predict_proba') else None
        if prediction[0] == 1:
            st.success(f'The model predicts that the person is at risk with a probability of {probability:.2f}.')
        else:
            st.info(f'The model predicts that the person is not at risk with a probability of {probability:.2f}.')
    else:
        st.error("Model or scaler not loaded properly.")

    # Plotting the input data
    st.markdown("---")
    st.header(f'Visualizing Your Input Data for {disease}')
    
    # Display input data as a bar chart
    input_df = pd.DataFrame([input_data[0]], columns=feature_names)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=input_df.columns, y=input_df.iloc[0])
    plt.title('Input Data Visualization')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    # Display distribution comparison if possible
    st.markdown("---")
    st.header(f'Feature Distribution Comparison for {disease}')
    
    data = datasets[disease]
    numeric_data = data.select_dtypes(include=[np.number])
    
    for feature in feature_names:
        plt.figure(figsize=(10, 4))
        sns.histplot(data[feature], kde=True, color='skyblue', label='Training Data Distribution')
        plt.axvline(input_data[0][feature_names.get_loc(feature)], color='red', linestyle='--', label='Input Value')
        plt.title(f'Distribution of {feature}')
        plt.legend()
        st.pyplot(plt)
