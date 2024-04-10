import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from data_preprocessing import DataProcessor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_pred

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, annot_kws={"size": 16})
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'images/training/{model_name}_confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_test, y_pred, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'images/training/{model_name}_roc_curve.png')
    plt.close()

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000),
            'params': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {'max_depth': [5, 10, 20], 'min_samples_leaf': [1, 2, 4]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
        }
    }

    best_model, best_score = None, 0
    results = []

    for name, data in models.items():
        print(f"\nTraining {name}...")
        grid_search = GridSearchCV(data['model'], data['params'], cv=5, scoring='f1')
        grid_search.fit(X_train_scaled, y_train)

        model = grid_search.best_estimator_
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1')
        mean_score = scores.mean()

        accuracy, precision, recall, f1, y_pred = evaluate_model(model, X_test_scaled, y_test)
        plot_confusion_matrix(y_test, y_pred, name)
        plot_roc_curve(y_test, y_pred, name)

        if f1 > best_score:
            best_model, best_score = model, f1

        results.append({
            'Model': name,
            'Best Params': grid_search.best_params_,
            'Cross-Val Score': mean_score,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    results_df = pd.DataFrame(results)
    print("\nModel Performance Summary:")
    print(results_df.to_string(index=False))

    return best_model, scaler

if __name__ == '__main__':
    data_path = 'data/raw/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()

    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()

    best_model, scaler = train_and_evaluate_models(X_train, y_train, X_test, y_test)

    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"\nBest model saved as 'models/best_model.pkl'")
    print(f"Scaler saved as 'models/scaler.pkl'")