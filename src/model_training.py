"""
Model Training Module for Breast Cancer Prediction

This module provides comprehensive model training and evaluation including:
- Multiple classifier comparison
- Hyperparameter tuning with GridSearchCV
- Cross-validation
- Comprehensive metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import joblib
from pathlib import Path
from data_preprocessing import DataProcessor
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # Calculate AUC if probabilities available
    if hasattr(model, 'predict_proba'):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics['auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return metrics


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: str = 'images/training'):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={'size': 18, 'weight': 'bold'},
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    
    # Add accuracy annotation
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray, model_name: str, output_dir: str = 'images/training'):
    """Create and save ROC curve visualization."""
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/{model_name.replace(" ", "_")}_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_test: np.ndarray, y_proba: np.ndarray, model_name: str, output_dir: str = 'images/training'):
    """Create and save Precision-Recall curve visualization."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='#3498db', lw=2.5, 
            label=f'PR Curve (AP = {avg_precision:.3f})')
    ax.fill_between(recall, precision, alpha=0.3, color='#3498db')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve: {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/{model_name.replace(" ", "_")}_pr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_df: pd.DataFrame, output_dir: str = 'images/training'):
    """Create comprehensive model comparison visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for ax, metric, color in zip(axes.flatten(), metrics, colors):
        bars = ax.barh(results_df['Model'], results_df[metric], color=color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.05)
        
        # Add value labels
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                   va='center', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_roc_curves(all_results: dict, y_test: np.ndarray, output_dir: str = 'images/training'):
    """Plot all ROC curves on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
    
    for (name, metrics), color in zip(all_results.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            roc_auc = metrics.get('auc', 0)
            ax.plot(metrics['fpr'], metrics['tpr'], color=color, lw=2, 
                   label=f'{name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/combined_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_and_evaluate_models(X_train: pd.DataFrame, y_train: pd.Series,
                               X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """
    Train and evaluate multiple models with hyperparameter tuning.
    
    Returns:
        Tuple of (best_model, scaler, results_df)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models and hyperparameters
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=10000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_leaf': [1, 2, 4, 8],
                'criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    }
    
    best_model = None
    best_f1 = 0
    results = []
    all_results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print('='*50)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=cv, 
            scoring='f1',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train_scaled, y_train)
        
        model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
        print(f"CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Evaluate on test set
        metrics = evaluate_model(model, X_test_scaled, y_test)
        all_results[name] = metrics
        
        # Print metrics
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            print(f"Test AUC: {metrics['auc']:.4f}")
        
        # Generate plots
        plot_confusion_matrix(y_test, metrics['y_pred'], name)
        if 'y_proba' in metrics:
            plot_roc_curve(y_test, metrics['y_proba'], name)
            plot_precision_recall_curve(y_test, metrics['y_proba'], name)
        
        # Track best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
        
        results.append({
            'Model': name,
            'Best Params': str(grid_search.best_params_),
            'CV F1 Mean': cv_scores.mean(),
            'CV F1 Std': cv_scores.std(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'AUC': metrics.get('auc', None)
        })
    
    results_df = pd.DataFrame(results)
    
    # Generate comparison plots
    plot_model_comparison(results_df)
    plot_combined_roc_curves(all_results, y_test)
    
    # Print summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']].to_string(index=False))
    print("\n" + "="*70)
    print(f"Best Model: {results_df.loc[results_df['F1 Score'].idxmax(), 'Model']}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print("="*70)
    
    # Save results
    results_df.to_csv('images/training/model_results.csv', index=False)
    
    return best_model, scaler, results_df


if __name__ == '__main__':
    # Ensure directories exist
    Path('images/training').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    data_path = 'data/raw/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()
    
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()
    
    # Train and evaluate models
    best_model, scaler = train_and_evaluate_models(X_train, y_train, X_test, y_test)[:2]
    
    # Save best model and scaler
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print(f"\nBest model saved to: models/best_model.pkl")
    print(f"Scaler saved to: models/scaler.pkl")