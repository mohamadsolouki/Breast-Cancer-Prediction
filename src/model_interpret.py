"""
Model Interpretation Module for Breast Cancer Prediction

This module provides comprehensive model interpretation including:
- Permutation importance
- SHAP values analysis
- Partial dependence plots
- Feature importance visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')


def permutation_importance_interpretation(model, X_test, y_test, feature_names, scaler, output_dir='images/interpretation'):
    """
    Calculate and visualize permutation importance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        scaler: Fitted scaler
        output_dir: Directory to save plots
    """
    from sklearn.inspection import permutation_importance
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    X_test_scaled = scaler.transform(X_test)
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    sorted_idx = result.importances_mean.argsort()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.boxplot(result.importances[sorted_idx].T, vert=False, 
               labels=[feature_names[i] for i in sorted_idx])
    ax.set_title("Permutation Feature Importance", fontsize=14, fontweight='bold')
    ax.set_xlabel("Decrease in Model Score", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: permutation_importance.png")


def shap_interpretation(model, X_train, X_test, feature_names, scaler, output_dir='images/interpretation'):
    """
    Calculate and visualize SHAP values.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        feature_names: List of feature names
        scaler: Fitted scaler
        output_dir: Directory to save plots
    """
    import shap
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)
    
    # Use appropriate explainer based on model type
    try:
        # Try TreeExplainer first for tree-based models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)
        
        # For binary classification, get values for positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        # Fall back to KernelExplainer for other models
        def predict_proba(X):
            return model.predict_proba(X)[:, 1]
        
        # Use a sample of training data for background
        background = shap.sample(X_train_transformed, min(100, len(X_train_transformed)))
        explainer = shap.KernelExplainer(predict_proba, background)
        shap_values = explainer.shap_values(X_test_transformed[:100])
        X_test_transformed = X_test_transformed[:100]
    
    # SHAP Summary (Beeswarm) Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: shap_beeswarm.png")
    
    # SHAP Bar Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, 
                     plot_type='bar', show=False)
    plt.title("SHAP Feature Importance (Bar)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: shap_bar.png")


def pdp_interpretation(model, X_train, feature_names, scaler, output_dir='images/interpretation'):
    """
    Create partial dependence plots.
    
    Args:
        model: Trained model
        X_train: Training features
        feature_names: List of feature names
        scaler: Fitted scaler
        output_dir: Directory to save plots
    """
    from sklearn.inspection import PartialDependenceDisplay
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    X_train_scaled = scaler.transform(X_train)
    
    # Select top features for PDP (to avoid overcrowded plots)
    n_features = min(10, len(feature_names))
    
    fig, ax = plt.subplots(figsize=(14, 24))
    
    try:
        PartialDependenceDisplay.from_estimator(
            model, X_train_scaled, 
            features=list(range(n_features)),
            feature_names=feature_names[:n_features],
            ax=ax,
            kind='average'
        )
        plt.suptitle("Partial Dependence Plots", fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pdp_plot.png', dpi=150, bbox_inches='tight')
        print("✓ Saved: pdp_plot.png")
    except Exception as e:
        print(f"Warning: Could not generate PDP plot: {e}")
    finally:
        plt.close()


def feature_importance(model, feature_names, output_dir='images/interpretation'):
    """
    Visualize model's built-in feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("Model does not have feature_importances_ or coef_ attribute")
            return
        
        # Sort by importance
        indices = np.argsort(importances)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(indices)))
        
        ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Model Feature Importance', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (idx, val) in enumerate(zip(indices, importances[indices])):
            ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: feature_importance.png")
        
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot: {e}")


def run_all_interpretations():
    """Run all model interpretation analyses."""
    from data_preprocessing import DataProcessor
    
    print("\n" + "="*60)
    print("MODEL INTERPRETATION ANALYSIS")
    print("="*60)
    
    # Create output directory
    output_dir = 'images/interpretation'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data and model
    processor = DataProcessor()
    processor.load_preprocessed_data(
        'data/processed/X_train.csv',
        'data/processed/X_test.csv',
        'data/processed/y_train.csv',
        'data/processed/y_test.csv'
    )
    
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()
    feature_names = processor.get_feature_names()
    
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    print(f"Model type: {type(model).__name__}")
    print(f"Features: {len(feature_names)}")
    print(f"Test samples: {len(X_test)}")
    
    print("\n" + "-"*60)
    print("Generating Visualizations...")
    print("-"*60)
    
    # Run interpretations
    feature_importance(model, feature_names, output_dir)
    permutation_importance_interpretation(model, X_test, y_test, feature_names, scaler, output_dir)
    shap_interpretation(model, X_train, X_test, feature_names, scaler, output_dir)
    pdp_interpretation(model, X_train, feature_names, scaler, output_dir)
    
    print("\n✅ Model interpretation complete!")
    print(f"📁 Visualizations saved to: {output_dir}")


if __name__ == '__main__':
    run_all_interpretations()