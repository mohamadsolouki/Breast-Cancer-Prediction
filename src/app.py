import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import DataProcessor
from PIL import Image
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


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

def explain_prediction(model, input_data, scaled_input_data, feature_names):
    # SHAP explanation
    explainer = shap.Explainer(model.predict, scaled_input_data)
    shap_values = explainer.shap_values(scaled_input_data)
    shap_explanation = shap.Explanation(shap_values, data=scaled_input_data, feature_names=feature_names)

    # LIME explanation
    explainer = LimeTabularExplainer(scaled_input_data, feature_names=feature_names, class_names=['Benign', 'Malignant'], discretize_continuous=True)
    exp = explainer.explain_instance(scaled_input_data[0], model.predict_proba, num_features=len(feature_names))
    lime_explanation = exp.as_list()

    return shap_explanation, lime_explanation

def main():
    st.set_page_config(page_title="Breast Cancer Prediction", page_icon=":female-doctor:")
    
    # Create navigation sidebar
    pages = ["Prediction", "Interpretation of model"]
    selected_page = st.sidebar.radio("Select a page", pages)
    
    if selected_page == "Prediction":
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

            # Explanation section
            st.subheader("Prediction Explanation")
            shap_explanation, lime_explanation = explain_prediction(model, input_data, scaled_input_data, feature_names)

            # SHAP explanation
            st.write("SHAP Explanation:")
            fig, ax = plt.subplots(figsize=(10, len(feature_names) * 0.4))
            shap.plots.bar(shap_explanation, show=False, ax=ax)
            ax.set_title("Feature Importance")
            ax.set_xlabel("SHAP Value")
            ax.set_ylabel("Feature")
            st.pyplot(fig)
            st.write("The SHAP values indicate the contribution of each feature to the model's prediction. Positive values push the prediction towards the positive class (Malignant), while negative values push towards the negative class (Benign).")

            # LIME explanation
            st.write("LIME Explanation:")
            lime_exp_df = pd.DataFrame(lime_explanation, columns=['Feature', 'Importance'])
            st.dataframe(lime_exp_df)
            st.write("The LIME explanation shows the contribution of each feature to the model's prediction for the specific instance. Positive values indicate features that push the prediction towards the positive class (Malignant), while negative values push towards the negative class (Benign).")
    
    elif selected_page == "Interpretation of model":
        st.title("Model Interpretation")

        # Correlation Heatmap
        correlation_heatmap = Image.open('images/interpretation/correlation_heatmap.png')
        st.image(correlation_heatmap, caption="Correlation Heatmap", use_column_width=True)
        st.write("The Correlation Heatmap shows the correlation between features in the dataset. Features with high correlation may have redundant information.")

        # SHAP Beeswarm Plot
        shap_beeswarm = Image.open('images/interpretation/shap_beeswarm.png')
        st.image(shap_beeswarm, caption="SHAP Beeswarm Plot", use_column_width=True)
        st.write("The SHAP Beeswarm plot shows the impact of each feature on the model's prediction. Features with high absolute SHAP values have a significant impact on the prediction.")

        # SHAP Bar Plot
        shap_bar = Image.open('images/interpretation/shap_bar.png')
        st.image(shap_bar, caption="SHAP Bar Plot", use_column_width=True)
        st.write("The SHAP Bar plot shows the average absolute SHAP values for each feature, indicating their overall importance in the model.")

        # Partial Dependence Plot
        pdp_plot = Image.open('images/interpretation/pdp_plot.png')
        st.image(pdp_plot, caption="Partial Dependence Plot", use_column_width=True)
        st.write("The Partial Dependence Plot shows the marginal effect of each feature on the predicted outcome. It helps understand how the model's predictions change as the feature values vary.")


if __name__ == '__main__':
    main()