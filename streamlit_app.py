import streamlit as st
from predict import predict_disease, load_average_values
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Multiple Disease Prediction App",
    layout="wide",
    page_icon="🏥"
)

# App Description
st.title("🏥 Multiple Disease Prediction App")
st.markdown("""
**Clinical Decision Support System** powered by machine learning for early risk assessment of:
- Cardiovascular Disease
- Diabetes Mellitus
- Parkinson's Disease
""")
st.markdown("---")

# Comprehensive Parameter Descriptions
PARAMETER_DESCRIPTIONS = {
    "Heart Disease": {
        "Age": "Patient age in years (29-77)",
        "Sex": "Biological sex (1 = male, 0 = female)",
        "ChestPainType": "1: Typical angina, 2: Atypical angina, 3: Non-anginal pain, 4: Asymptomatic",
        "RestingBP": "Resting blood pressure in mm Hg (94-200)",
        "Cholesterol": "Serum cholesterol in mg/dL (126-564)",
        "FastingBS": "Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)",
        "RestingECG": "0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy",
        "MaxHR": "Maximum heart rate achieved (71-202 bpm)",
        "ExerciseAngina": "Exercise-induced angina (1 = yes, 0 = no)",
        "Oldpeak": "ST depression induced by exercise relative to rest (0-6.2)",
        "ST_Slope": "Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)",
        "Ca": "Number of major vessels (0-3) colored by fluoroscopy",
        "Thal": "1: Normal, 2: Fixed defect, 3: Reversible defect"
    },
    "Diabetes": {
        "Pregnancies": "Number of times pregnant (0-17)",
        "Glucose": "Plasma glucose concentration (0-199 mg/dL)",
        "BloodPressure": "Diastolic blood pressure (0-122 mm Hg)",
        "SkinThickness": "Triceps skinfold thickness (0-99 mm)",
        "Insulin": "2-Hour serum insulin (0-846 μU/mL)",
        "BMI": "Body mass index (0-67.1 kg/m²)",
        "DiabetesPedigreeFunction": "Diabetes likelihood genetic score (0.08-2.42)",
        "Age": "Age in years (21-81)"
    },
    "Parkinson's": {
        "MDVP:Fo(Hz)": "Average vocal fundamental frequency (88-260 Hz)",
        "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency (102-592 Hz)",
        "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency (65-239 Hz)",
        "MDVP:Jitter(%)": "Cycle-to-cycle variation in frequency (0.001-0.033)",
        "MDVP:Jitter(Abs)": "Absolute jitter measure (0.000007-0.0002)",
        "MDVP:RAP": "Relative amplitude perturbation (0.0006-0.021)",
        "MDVP:PPQ": "Five-point period perturbation quotient (0.0008-0.02)",
        "Jitter:DDP": "Average absolute difference of differences between cycles (0.002-0.064)",
        "MDVP:Shimmer": "Amplitude variation (0.009-0.119)",
        "MDVP:Shimmer(dB)": "Shimmer in decibels (0.085-1.302 dB)",
        "Shimmer:APQ3": "3-point amplitude perturbation quotient (0.004-0.056)",
        "Shimmer:APQ5": "5-point amplitude perturbation quotient (0.005-0.079)",
        "MDVP:APQ": "Amplitude perturbation quotient (0.007-0.138)",
        "Shimmer:DDA": "Average absolute differences between amplitudes (0.013-0.169)",
        "NHR": "Noise-to-harmonics ratio (0.0006-0.314)",
        "HNR": "Harmonics-to-noise ratio (8.44-33.047 dB)",
        "RPDE": "Recurrence period density entropy (0.256-0.685)",
        "DFA": "Detrended fluctuation analysis (0.574-0.825)",
        "spread1": "Nonlinear measure of fundamental frequency variation (-7.96-0.209)",
        "spread2": "Nonlinear measure of fundamental frequency variation (0.006-0.45)",
        "D2": "Correlation dimension (1.42-3.671)",
        "PPE": "Pitch period entropy (0.044-0.527)"
    }
}

# Feature sets
feature_sets = {
    disease: list(params.keys()) for disease, params in PARAMETER_DESCRIPTIONS.items()
}

# Disease Selection
disease = st.selectbox(
    "Select Disease for Assessment", 
    list(PARAMETER_DESCRIPTIONS.keys()),
    help="Choose the condition for risk evaluation"
)

# Input Form
st.header(f"Patient Parameters for {disease}")
with st.expander("📋 Data Entry Guidelines"):
    st.markdown("""
    1. Enter **precise clinical measurements** from patient tests
    2. For categorical parameters, use the specified numeric codes
    3. All fields must contain valid values (0 is used as default)
    4. Reference ranges shown in parameter tooltips
    """)

with st.form("clinical_form"):
    cols = st.columns(3)
    user_input = {}
    
    for i, feature in enumerate(feature_sets[disease]):
        with cols[i % 3]:
            user_input[feature] = st.number_input(
                label=f"{feature}",
                value=0.0,
                min_value=0.0,
                step=0.1 if any(x in feature.lower() for x in ['jitter', 'shimmer', 'ppq', 'apq']) else 1.0,
                format="%.3f" if "Jitter" in feature else "%.1f",
                help=PARAMETER_DESCRIPTIONS[disease][feature]
            )
    
    submitted = st.form_submit_button("🔬 Analyze Clinical Risk")

# Prediction and Results
if submitted:
    if all(v == 0 for v in user_input.values()):
        st.warning("⚠️ Using default values - enter actual patient data for accurate assessment")
    else:
        with st.spinner("Processing clinical biomarkers..."):
            prediction = predict_disease(disease, user_input)
            
            # Display prediction
            if "Positive" in prediction:
                st.error(f"## 🚨 Clinical Alert: {prediction}")
                st.warning("Recommendation: Immediate specialist consultation advised")
            else:
                st.success(f"## ✅ Negative Screening: {prediction}")
                st.info("Recommendation: Routine monitoring suggested")
            
            # Clinical comparison
            st.markdown("---")
            st.header("Biomarker Comparison")
            avg_values = load_average_values(disease)
            
            if avg_values:
                comparison_data = pd.DataFrame({
                    "Biomarker": list(user_input.keys()),
                    "Patient Value": list(user_input.values()),
                    "Population Norm": [avg_values.get(k, 0) for k in user_input.keys()],
                    "Reference Range": [PARAMETER_DESCRIPTIONS[disease][k].split("(")[-1].replace(")","") 
                                     for k in user_input.keys()]
                })
                
                # Visual comparison
                st.bar_chart(
                    comparison_data.set_index("Biomarker")[["Patient Value", "Population Norm"]],
                    color=["#FF0000", "#0068C9"]
                )
                
                # Clinical data table
                st.dataframe(
                    comparison_data,
                    column_config={
                        "Biomarker": "Clinical Parameter",
                        "Patient Value": st.column_config.NumberColumn("Patient Value"),
                        "Population Norm": st.column_config.NumberColumn("Population Average"),
                        "Reference Range": "Expected Range"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("Population reference data unavailable for comparison")

# Clinical Footer
st.markdown("---")
st.caption("""
**Clinical Disclaimer**: This AI-powered tool provides preliminary risk assessment only. 
Results require validation by qualified healthcare professionals. 
Not intended for diagnostic use without clinical correlation.
""")
