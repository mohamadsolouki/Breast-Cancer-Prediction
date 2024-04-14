import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay
from data_preprocessing import DataProcessor
import joblib
import numpy as np
from sklearn.inspection import permutation_importance

# Function to interpret the model using Permutation Importance
def permutation_importance_interpretation(model, X_test, y_test, feature_names, scaler):
    result = permutation_importance(model, scaler.transform(X_test), y_test, n_repeats=10, random_state=0)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[i] for i in sorted_idx])
    ax.set_title("Permutation Importances")
    plt.tight_layout()
    plt.savefig('images/interpretation/permutation_importance.png')
    plt.close()

# Function to visualize the distribution of features in the training data
def feature_distribution(X_train, y_train, feature_names, scaler):
    X_train_unscaled = pd.DataFrame(scaler.inverse_transform(X_train), columns=feature_names)
    data = pd.concat([X_train_unscaled, y_train], axis=1)
    data.columns = feature_names + ['diagnosis']
    
    for feature in feature_names:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x=feature, kde=True, hue='diagnosis')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend(title="Diagnosis", labels=["Benign", "Malignant"])
        plt.savefig(f'images/feature_distribution/dist_{feature}.png')
        plt.close()


# Function to interpret the model using SHAP values
def shap_interpretation(model, X_train, X_test, feature_names, scaler):
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    # Ensure training data is transformed
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Create SHAP explainer using the model and the transformed training data
    explainer = shap.Explainer(predict_fn, X_train_transformed)
    shap_values = explainer(X_test_transformed)

    # Set the feature names directly in the SHAP values data structure
    shap_values.feature_names = feature_names

    # SHAP summary plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values.values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig('images/interpretation/shap_beeswarm.png')
    plt.close()

    # SHAP bar plot
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_values, max_display=len(feature_names), show=False)
    plt.title("SHAP Bar Plot")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig('images/interpretation/shap_bar.png')
    plt.close()


# Function to interpret the model using PDP
def pdp_interpretation(model, X_train, feature_names, scaler):
    fig, ax = plt.subplots(figsize=(12, 30))
    PartialDependenceDisplay.from_estimator(model, scaler.transform(X_train), features=range(len(feature_names)), feature_names=feature_names, ax=ax)
    plt.tight_layout()
    plt.savefig('images/interpretation/pdp_plot.png')
    plt.close()


# Function to visualize feature importance 
def feature_importance(model, feature_names):
    try:
        importances = model.feature_importances_
    except AttributeError:
        if hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            raise AttributeError("The model does not have 'feature_importances_' or 'coef_' attribute.")
    
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 10))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('images/interpretation/feature_importance.png')
    plt.close()

# Main function to interpret the model
if __name__ == '__main__':
    X_train_path = 'data/processed/X_train.csv'
    X_test_path = 'data/processed/X_test.csv'
    y_train_path = 'data/processed/y_train.csv'
    y_test_path = 'data/processed/y_test.csv'

    processor = DataProcessor()
    processor.load_preprocessed_data(X_train_path, X_test_path, y_train_path, y_test_path)
    feature_names = processor.feature_names(X_train_path)
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()

    feature_distribution(X_train, y_train, feature_names, scaler)
    permutation_importance_interpretation(model, X_test, y_test, feature_names, scaler)
    shap_interpretation(model, X_train, X_test, feature_names, scaler)
    pdp_interpretation(model, X_train, feature_names, scaler)
    feature_importance(model, feature_names)