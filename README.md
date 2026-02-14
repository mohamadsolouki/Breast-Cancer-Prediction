
# Breast Cancer Prediction Application

## Overview

A comprehensive machine learning application for breast cancer diagnosis prediction using the Wisconsin Breast Cancer (Diagnostic) dataset. This application predicts whether a breast tumor is benign or malignant based on diagnostic measurements derived from Fine Needle Aspiration (FNA) image data.

Built with interpretability and clinical usability in mind, the application provides detailed explanations for predictions to aid oncologists and medical practitioners in decision-making processes.

## Key Features

- **Interactive Prediction Interface**: User-friendly Streamlit web application with real-time predictions
- **Multiple ML Models**: Comparison of Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and SVM
- **Comprehensive EDA**: Automated exploratory data analysis with visualization generation
- **Model Interpretability**: SHAP values, permutation importance, and partial dependence plots
- **Visual Analytics**: Interactive Plotly visualizations including radar charts and gauge indicators
- **Best Practices**: Stratified sampling, cross-validation, hyperparameter tuning, and proper data scaling

## Project Structure

```
Breast-Cancer-Prediction/
│
├── app/
│   └── app.py                    # Enhanced Streamlit application with multiple pages
│
├── data/
│   ├── raw/                      # Original Wisconsin Breast Cancer dataset
│   │   └── data.csv
│   └── processed/                # Train/test split data
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── images/
│   ├── app/                      # Runtime-generated visualizations
│   ├── EDA/                      # Exploratory data analysis plots
│   │   └── feature_distribution/ # Per-feature distribution plots
│   ├── interpretation/           # Model interpretation visualizations
│   └── training/                 # Model training and evaluation plots
│
├── models/
│   ├── best_model.pkl            # Best performing trained model
│   └── scaler.pkl                # Fitted StandardScaler
│
├── notebooks/
│   └── EDA.ipynb                 # Interactive exploratory analysis notebook
│
├── src/
│   ├── data_preprocessing.py     # Data loading, cleaning, and splitting
│   ├── eda.py                    # Comprehensive EDA and visualization generation
│   ├── model_training.py         # Model training with cross-validation and tuning
│   └── model_interpret.py        # SHAP, permutation importance, PDP analysis
│
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md
```

## Dataset

The **Wisconsin Breast Cancer (Diagnostic)** dataset contains 569 samples with 30 features computed from digitized images of FNA of breast masses.

### Features
For each cell nucleus, ten characteristics are measured:
- **Radius**: Mean distance from center to perimeter points
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**: Perimeter of the nucleus
- **Area**: Area of the nucleus
- **Smoothness**: Local variation in radius lengths
- **Compactness**: Perimeter² / Area - 1.0
- **Concavity**: Severity of concave portions
- **Concave Points**: Number of concave portions
- **Symmetry**: Symmetry of the nucleus
- **Fractal Dimension**: "Coastline approximation" - 1

Each feature is computed as:
- **Mean**: Average across all cells
- **SE**: Standard error
- **Worst**: Largest (mean of three largest values)

### Class Distribution
- **Benign (B)**: 357 samples (62.7%)
- **Malignant (M)**: 212 samples (37.3%)

> **Citation**: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mohamadsolouki/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Generate EDA Visualizations
```bash
python src/eda.py
```
This creates comprehensive visualizations in `images/EDA/`.

### 2. Train Models
```bash
python src/model_training.py
```
Trains multiple models, performs hyperparameter tuning, and saves the best model.

### 3. Generate Model Interpretations
```bash
python src/model_interpret.py
```
Creates SHAP plots, permutation importance, and partial dependence plots.

### 4. Run the Application
```bash
streamlit run app/app.py
```
Navigate to `http://localhost:8501` in your browser.

## Application Pages

### 🔬 Prediction
- Input patient FNA measurements
- Get real-time malignancy predictions with confidence scores
- View interactive gauge charts and probability distributions
- Compare patient values against class averages with radar charts

### 📊 Data Analysis
- Explore dataset statistics and class distributions
- Interactive feature distribution visualizations
- Correlation heatmaps and scatter matrices

### 📈 Model Performance
- Compare all trained models side-by-side
- View accuracy, precision, recall, F1, and AUC metrics
- Examine confusion matrices and ROC curves

### 🔍 Model Interpretation
- SHAP beeswarm and bar plots for feature importance
- Permutation importance analysis
- Partial dependence plots
- Feature correlation analysis

### ℹ️ About
- Dataset documentation
- Feature descriptions
- Medical disclaimer

## Model Performance

The application trains and evaluates multiple models:

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | ~0.97 | ~0.96 | ~0.95 | ~0.96 | ~0.99 |
| Random Forest | ~0.96 | ~0.95 | ~0.94 | ~0.95 | ~0.99 |
| Gradient Boosting | ~0.97 | ~0.96 | ~0.95 | ~0.96 | ~0.99 |
| SVM | ~0.97 | ~0.97 | ~0.95 | ~0.96 | ~0.99 |
| Decision Tree | ~0.93 | ~0.90 | ~0.90 | ~0.90 | ~0.90 |

*Note: Actual values may vary based on random state.*

## Technologies Used

- **Python 3.10+**
- **Streamlit**: Interactive web application
- **Scikit-learn**: Machine learning models and preprocessing
- **Plotly**: Interactive visualizations
- **SHAP**: Model interpretability
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Static visualizations

## Disclaimer

⚠️ **This application is for educational and research purposes only.** It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## Acknowledgments

- UCI Machine Learning Repository for the Wisconsin Breast Cancer dataset
- The scikit-learn and Streamlit communities
