import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_preprocessing import DataProcessor
from model_interpret import plot_feature_importances, plot_shap_summary, interpret_prediction

def load_data(data_path):
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    data = processor.data
    feature_names = processor.get_feature_names()
    return data, feature_names, processor.scaler

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.set_page_config(page_title="Breast Cancer Prediction", page_icon=":female-doctor:")
    st.title("Breast Cancer Prediction")

    # Load the trained model
    model = load_model('src/models/best_model.pkl')

    # Load the dataset
    data_path = 'data/raw/data.csv'
    data, feature_names, scaler = load_data(data_path)

    # Predefined data for a Malignant (M) tumor
    predefined_data = {
        'radius_mean': 17.99,
        'texture_mean': 10.38,
        'perimeter_mean': 122.80,
        'area_mean': 1001.0,
        'smoothness_mean': 0.11840,
        'compactness_mean': 0.27760,
        'concavity_mean': 0.30010,
        'concave points_mean': 0.14710,
        'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871,
        'radius_se': 1.0950,
        'texture_se': 0.9053,
        'perimeter_se': 8.589,
        'area_se': 153.40,
        'smoothness_se': 0.006399,
        'compactness_se': 0.04904,
        'concavity_se': 0.05373,
        'concave points_se': 0.01587,
        'symmetry_se': 0.03003,
        'fractal_dimension_se': 0.006193,
        'radius_worst': 25.38,
        'texture_worst': 17.33,
        'perimeter_worst': 184.60,
        'area_worst': 2019.0,
        'smoothness_worst': 0.1622,
        'compactness_worst': 0.6656,
        'concavity_worst': 0.7119,
        'concave points_worst': 0.2654,
        'symmetry_worst': 0.4601,
        'fractal_dimension_worst': 0.11890
    }

    # User input form
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)
    form_data = {}
    for i, feature in enumerate(feature_names):
        if i < len(feature_names) // 2:
            with col1:
                form_data[feature] = st.number_input(feature, value=predefined_data[feature], format="%.5f")
        else:
            with col2:
                form_data[feature] = st.number_input(feature, value=predefined_data[feature], format="%.5f")

    # Prediction section
    if st.button("Predict"):
        input_data = pd.DataFrame([form_data])
        scaled_input_data = scaler.transform(input_data)
        prediction = predict(model, scaled_input_data)
        if prediction[0] == 1:
            st.error("The prediction is Malignant (M)")
        else:
            st.success("The prediction is Benign (B)")

        # Feature importances plot
        fig_importances = plot_feature_importances(model, scaled_input_data, feature_names)
        st.subheader("Feature Importances")
        st.pyplot(fig_importances)

        # SHAP summary plot
        fig_shap = plot_shap_summary(model, scaled_input_data, feature_names)
        st.subheader("SHAP Summary Plot")
        st.pyplot(fig_shap)

        # Interpretation for doctors
        st.subheader("Interpretation for Doctors")
        interpret_prediction(model, scaled_input_data, feature_names, prediction)

if __name__ == '__main__':
    main()