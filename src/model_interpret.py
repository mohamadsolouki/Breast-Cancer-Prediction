import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
from sklearn.inspection import PartialDependenceDisplay
from data_preprocessing import DataProcessor
import joblib

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

def correlation_analysis(X_train, feature_names, scaler):
    X_train_unscaled = pd.DataFrame(scaler.inverse_transform(X_train), columns=feature_names)
    plt.figure(figsize=(20, 20))
    sns.heatmap(X_train_unscaled.corr(), annot=False, cmap='coolwarm', fmt='.2f', cbar_kws={"shrink": .9})
    plt.title("Correlation Heatmap")
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig('images/interpretation/correlation_heatmap.png')
    plt.close()

def shap_interpretation(model, X_train, X_test, feature_names, scaler):
    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.Explainer(predict_fn, scaler.transform(X_train))
    shap_values = explainer(scaler.transform(X_test))

    plt.figure(figsize=(12, 10))
    shap.plots.beeswarm(shap_values, max_display=len(feature_names), show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.tight_layout()
    plt.savefig('images/interpretation/shap_beeswarm.png')
    plt.close()

    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_values, max_display=len(feature_names), show=False)
    plt.title("SHAP Bar Plot")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.yticks(range(len(feature_names)), feature_names)
    plt.tight_layout()
    plt.savefig('images/interpretation/shap_bar.png')
    plt.close()

def lime_interpretation(model, X_train, feature_names, class_names, scaler):
    explainer = lime_tabular.LimeTabularExplainer(scaler.transform(X_train), feature_names=feature_names, class_names=class_names, discretize_continuous=True)
    exp = explainer.explain_instance(scaler.transform(X_train)[0], model.predict_proba, num_features=len(feature_names))
    exp.save_to_file('images/interpretation/lime_explanation.html')

def pdp_interpretation(model, X_train, feature_names, scaler):
    fig, ax = plt.subplots(figsize=(12, 30))
    PartialDependenceDisplay.from_estimator(model, scaler.transform(X_train), features=range(len(feature_names)), feature_names=feature_names, ax=ax)
    plt.tight_layout()
    plt.savefig('images/interpretation/pdp_plot.png')
    plt.close()

if __name__ == '__main__':
    X_train_path = 'data/processed/X_train.csv'
    X_test_path = 'data/processed/X_test.csv'
    y_train_path = 'data/processed/y_train.csv'
    y_test_path = 'data/processed/y_test.csv'

    processor = DataProcessor()
    processor.load_preprocessed_data(X_train_path, X_test_path, y_train_path, y_test_path)
    feature_names = pd.read_csv(X_train_path, nrows=0).columns.tolist()

    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()

    feature_distribution(X_train, y_train, feature_names, scaler)
    correlation_analysis(X_train, feature_names, scaler)

    shap_interpretation(model, X_train, X_test, feature_names, scaler)
    lime_interpretation(model, X_train, feature_names, ['Benign', 'Malignant'], scaler)
    pdp_interpretation(model, X_train, feature_names, scaler)