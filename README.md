
# Breast Cancer Prediction Application

## Overview

This repository hosts the code and documentation for a breast cancer prediction application developed using Python and Streamlit. The application predicts whether a breast tumor is benign or malignant based on diagnostic measurements derived from image data. It is designed to be not only accurate but also interpretable, providing explanations for its predictions to aid oncologists and medical practitioners in decision-making processes.

## Project Structure

```
Breast-Cancer-Prediction/
│
├── data/
│   ├── raw/                  # Contains the original, unprocessed datasets
│   └── processed/            # Contains processed data that is ready for modeling
│
├── models/                   # Trained model objects and scaler files
│   ├── best_model.pkl        # The best performing model saved as a pickle file
│   └── scaler.pkl            # Scaler object for normalizing input features
│
├── src/                      # Source files for the project
│   ├── data_preprocessing.py # Script for data cleaning and preprocessing
│   ├── model_training.py     # Script for training machine learning models
│   └── model_interpret.py    # Script for interpreting models using various techniques
│
├── notebooks/                # Jupyter notebooks for exploratory data analysis
│   └── EDA.ipynb             # Notebook containing exploratory data analysis
│
├── images/                   # Contains generated plots and figures for documentation
│   ├── app/
│   └── EDA/
│   └── interpretation/
│   └── training/          
│
├── app/                      # Streamlit application files
│   └── app.py                # Main application script for running the Streamlit interface
│
└── README.md                 # Project README file
```

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains features computed from digitized images of FNA samples. The dataset includes 569 instances, each with 30 features and a binary target variable indicating the diagnosis (M = malignant, B = benign).
Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Features

- **Data Preprocessing**: Cleansing and normalization of image-derived features.
- **Model Training**: Utilizes Logistic Regression, Decision Trees, Random Forest, and SVM. Includes hyperparameter tuning and cross-validation.
- **Model Interpretation**: Implements SHAP, LIME, and permutation importance for explaining model predictions.
- **Interactive Application**: Built with Streamlit, offering a user-friendly interface for making predictions and understanding model outputs.
- **Visualization**: Generates and stores visual explanations including feature importance, partial dependence plots, and correlation heatmaps.
- **Interpretability**: Focuses on model interpretability and transparency to aid medical professionals in understanding predictions.

## Installation

To set up the project environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/mohamadsolouki/Breast-Cancer-Prediction.git
   cd Breast-Cancer-Prediction
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Models and Training

Model training scripts are located in the `src` directory. To train the models and evaluate their performance, run:

```bash
python src/model_training.py
```
This script will output the performance of the models and save the best model to the `models` directory.


## Usage

It is better to first run the python scripts in the `src` directory from the root directory of the project. For example, to preprocess the data, run:
```bash
python src/data_preprocessing.py
```

To run the Streamlit application:

```bash
streamlit run app/app.py
```

Navigate to `http://localhost:8501` in your web browser to view the application.



## License

This project is licensed under the [MIT License](LICENSE).
