import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import shap
import streamlit as st

def plot_feature_importances(model, feature_names):
    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
        indices = np.argsort(np.abs(coefficients))[::-1]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Feature Importances")
        ax.bar(range(len(feature_names)), coefficients[indices], align="center")
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
        ax.set_xlim([-1, len(feature_names)])
        fig.tight_layout()
        return fig
    else:
        raise ValueError("Model does not have coefficients for feature importances.")

def plot_shap_summary(model, input_data, feature_names):
    explainer = shap.Explainer(model, input_data)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, input_data, feature_names=feature_names, plot_type="bar", show=False)
    return fig

def interpret_prediction(model, input_data, feature_names, prediction):
    if prediction[0] == 1:
        st.write("Based on the provided patient information, the model predicts that the breast tumor is likely to be Malignant (M). The key factors contributing to this prediction are:")
        shap_values = shap.LinearExplainer(model, input_data).shap_values(input_data)
        top_features = np.argsort(np.abs(shap_values[0]))[::-1][:3]
        for feature in top_features:
            st.write(f"- {feature_names[feature]}: This feature has a significant impact on the prediction of Malignant (M) tumors.")
    else:
        st.write("Based on the provided patient information, the model predicts that the breast tumor is likely to be Benign (B). The key factors contributing to this prediction are:")
        shap_values = shap.LinearExplainer(model, input_data).shap_values(input_data)
        top_features = np.argsort(np.abs(shap_values[0]))[::-1][:3]
        for feature in top_features:
            st.write(f"- {feature_names[feature]}: This feature has a significant impact on the prediction of Benign (B) tumors.")