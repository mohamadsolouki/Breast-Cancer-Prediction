import streamlit as st
import pandas as pd
import joblib
from PIL import Image
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import sys
sys.path.append('src/')
from data_preprocessing import DataProcessor


def load_data(data_path):
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    data = processor.data
    feature_names = processor.get_feature_names()
    scaler = joblib.load('models/scaler.pkl')
    return data, feature_names, scaler

def load_preprocessed_data(X_train_path, X_test_path, y_train_path, y_test_path):
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)['diagnosis'].values
    y_test = pd.read_csv(y_test_path)['diagnosis'].values
    return X_train, X_test, y_train, y_test

def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict(model, input_data, scaler):
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return prediction


def lime_interpretation(model, X_train, feature_names, class_names, scaler):
    explainer = lime_tabular.LimeTabularExplainer(scaler.transform(X_train), feature_names=feature_names, class_names=class_names, discretize_continuous=True)
    exp = explainer.explain_instance(scaler.transform(X_train)[0], model.predict_proba, num_features=len(feature_names))
    exp.save_to_file('images/interpretation/lime_prediction_explanation.html')


def main():
    st.set_page_config(page_title="Breast Cancer Prediction", page_icon=":female-doctor:")
    
    # Create navigation sidebar
    pages = ["Prediction", "Interpretation of model"]
    selected_page = st.sidebar.radio("Select a page", pages)
    
    if selected_page == "Prediction":
        st.title("Breast Cancer Prediction")
        
        # Load the trained model
        model = load_model('models/best_model.pkl')
        
        # Load the dataset
        data_path = 'data/raw/data.csv'
        data, feature_names, scaler = load_data(data_path)

        # Load preprocessed data
        X_train_path = 'data/processed/X_train.csv'
        X_test_path = 'data/processed/X_test.csv'
        y_train_path = 'data/processed/y_train.csv'
        y_test_path = 'data/processed/y_test.csv'
        X_train, X_test, y_train, y_test = load_preprocessed_data(X_train_path, X_test_path, y_train_path, y_test_path)
        
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
        prediction = predict(model, input_data, scaler)
        if prediction[0] == 1:
            st.error("The prediction is Malignant (M)")
        else:
            st.success("The prediction is Benign (B)")

        # Prediction probability
        st.write("Prediction Probability:")
        prediction_proba = model.predict_proba(scaler.transform(input_data))[0]
        st.write(f"Benign (B): {prediction_proba[0]:.5f}")
        st.write(f"Malignant (M): {prediction_proba[1]:.5f}")

        # Explanation section
        st.subheader("Prediction Explanation")
        st.write("The LIME explanation provides a local interpretation of the model's prediction for the input data.")
        st.write("It shows the contribution of each feature to the prediction.")
        st.write("Lime explanation is a local surrogate model that explains the model's prediction for a single instance.")
        st.write("The values shown in the LIME explanations are scaled to the original feature values because the model was trained on scaled data for better performance.")


        # Create a LimeTabularExplainer
        explainer = LimeTabularExplainer(scaler.transform(data.drop('diagnosis', axis=1).values), 
                                        feature_names=feature_names, 
                                        class_names=['Benign', 'Malignant'], 
                                        discretize_continuous=True)

        # Explain the model's prediction
        exp = explainer.explain_instance(scaler.transform(input_data)[0], model.predict_proba, num_features=len(feature_names))

        # Save the explanation as an HTML file
        exp.save_to_file('images/app/lime_explanation.html')

        # Display the explanation
        st.write("LIME Explanation:")
        components.html(open('images/app/lime_explanation.html', 'r', encoding='utf-8').read(), height=2000)


    elif selected_page == "Interpretation of model":
        st.title("Model Interpretation")

        # Feature Distribution
        st.subheader("Feature Distribution")
        feature_distribution = Image.open('images/feature_distribution/dist_radius_mean.png')
        st.image(feature_distribution, caption="Feature Distribution", use_column_width=True)
        st.write("The Sample Feature Distribution plot shows the distribution of radius_mean feature for both Benign and Malignant tumors.")

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        correlation_heatmap = Image.open('images/EDA/correlation_heatmap.png')
        st.image(correlation_heatmap, caption="Correlation Heatmap", use_column_width=True)
        st.write("The Correlation Heatmap shows the correlation between features in the dataset. Features with high correlation may have redundant information.")

        # SHAP Beeswarm Plot
        st.subheader("SHAP Plots")
        shap_beeswarm = Image.open('images/interpretation/shap_beeswarm.png')
        st.image(shap_beeswarm, caption="SHAP Beeswarm Plot", use_column_width=True)
        st.write("The SHAP Beeswarm plot shows the impact of each feature on the model's prediction. Features with high absolute SHAP values have a significant impact on the prediction.")

        # SHAP Bar Plot
        shap_bar = Image.open('images/interpretation/shap_bar.png')
        st.image(shap_bar, caption="SHAP Bar Plot", use_column_width=True)
        st.write("The SHAP Bar plot shows the average absolute SHAP values for each feature, indicating their overall importance in the model.")

        # Partial Dependence Plot
        st.subheader("Partial Dependence Plot")
        pdp_plot = Image.open('images/interpretation/pdp_plot.png')
        st.image(pdp_plot, caption="Partial Dependence Plot", use_column_width=True)
        st.write("The Partial Dependence Plot shows the marginal effect of each feature on the predicted outcome. It helps understand how the model's predictions change as the feature values vary.")


if __name__ == '__main__':
    
    # Increase the pixel limit
    Image.MAX_IMAGE_PIXELS = None
    
    main()