import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def interpret_prediction_with_shap(model, input_data, data):
    st.markdown("### SHAP Prediction Interpretation")

    # Calculate SHAP values using the model and the input_data
    explainer = shap.Explainer(model, data)
    shap_values = explainer(input_data)

    # Get the expected value from the explainer, required for the force plot
    expected_value = explainer.expected_value

    # Generate force plot
    # force plot for a single prediction, thus we use shap_values[0]
    shap.plots.force(base_value=expected_value, shap_values=shap_values.values[0], feature_names=feature_names)

    # If you want to show the plot in the Streamlit app, use shap's matplotlib=True
    # and Streamlit's st.pyplot() to render the plot.
    shap.plots.force(base_value=expected_value, shap_values=shap_values.values[0], feature_names=feature_names, matplotlib=True)
    st.pyplot(bbox_inches='tight')
    plt.clf()  # Clear the current figure after rendering it in Streamlit


def interpret_model_coefficients(model, feature_names):
    """
    Interprets the coefficients of a logistic regression model.

    Parameters:
    - model: A trained logistic regression model.
    - feature_names: A list of feature names.

    Returns:
    - A Streamlit component that displays the model coefficients in an interpretable manner.
    """
    st.markdown("### Model Coefficients Interpretation")
    st.write("The contribution of each feature to the prediction can be understood by examining the model's coefficients. Positive coefficients increase the log-odds of the prediction being malignant, while negative coefficients decrease it.")

    coeffs = model.coef_[0]
    coeff_dict = {feature_names[i]: coeffs[i] for i in range(len(feature_names))}

    # Sort features by the absolute value of their coefficient
    sorted_features = sorted(coeff_dict.items(), key=lambda kv: np.abs(kv[1]), reverse=True)

    for feature, coeff in sorted_features:
        st.text(f"{feature}: {'+' if coeff > 0 else ''}{coeff:.4f}")