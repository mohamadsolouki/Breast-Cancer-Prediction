import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import joblib
from data_preprocessing import DataProcessor
from tabulate import tabulate

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5, 10]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10]
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }
    }

    best_model = None
    best_score = 0
    best_model_name = ''

    results = []

    for model_name, model_data in models.items():
        model = model_data['model']
        params = model_data['params']

        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5)
        grid_search.fit(X_train, y_train)
        best_model_params = grid_search.best_params_

        model.set_params(**best_model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_model_name = model_name

        results.append({
            'Model': model_name,
            'Best Parameters': str(best_model_params),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Cross-validation Mean': cv_mean,
            'Cross-validation Std': cv_std
        })

    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Best Parameters', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Cross-validation Mean', 'Cross-validation Std']]
    results_table = tabulate(results_df, headers='keys', tablefmt='grid', showindex=False)

    print("\nModel Performance Results:")
    print(results_table)

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Cross-validation Score: {best_score:.4f}")

    return best_model

def plot_feature_importances(model, X_test, y_test, feature_names):
    importances = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances = importances.importances_mean
    std = importances.importances_std

    indices = np.argsort(feature_importances)[::-1]
    feature_names_sorted = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), feature_importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(feature_names)), feature_names_sorted, rotation=90)
    plt.xlim([-1, len(feature_names)])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    data_path = 'data/raw/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()

    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()
    feature_names = processor.get_feature_names()

    best_model = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    joblib.dump(best_model, 'best_model.pkl')
    print("\nBest model saved as 'best_model.pkl'")

    print("\nFeature Importances:")
    plot_feature_importances(best_model, X_test, y_test, feature_names)