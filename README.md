# Breast Cancer Prediction App

This project aims to develop an interpretable model for breast cancer prediction using machine learning techniques. The model predicts whether a tumor is malignant or benign based on various features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, which contains features computed from digitized images of FNA samples. The dataset includes 569 instances, each with 30 features and a binary target variable indicating the diagnosis (M = malignant, B = benign).

## Project Structure

- `data/`: Contains the dataset file (`data.csv`).
- `src/`: Contains the source code files for data preprocessing, model training, evaluation, and interpretation.
- `notebooks/`: Contains Jupyter notebooks for exploratory data analysis.
- `requirements.txt`: Lists the required Python dependencies for the project.
- `README.md`: Provides an overview of the project and instructions for setup and usage.

## Setup and Usage

1. Clone the repository:
git clone https://github.com/your-username/breast-cancer-prediction.git


2. Install the required dependencies:
pip install -r requirements.txt


3. Run the Streamlit app:
streamlit run src/app.py

angelscript
Copy

4. Access the app in your web browser at `http://localhost:8501`.

## Results and Interpretation

The project explores various machine learning models, including Random Forest, Support Vector Machine, and Logistic Regression, for breast cancer prediction. The models are trained and evaluated using appropriate performance metrics such as accuracy, precision, recall, and F1 score.

The interpretability of the models is investigated through feature importance analysis and permutation importance. The results provide insights into the most significant features contributing to the predictions.

## Future Improvements

- Implement a user-friendly interface for inputting new data points and obtaining predictions.
- Explore additional interpretability techniques such as SHAP values or LIME.
- Investigate the potential of deep learning models for improved performance.
- Conduct thorough error analysis to understand the model's limitations and identify areas for improvement.

## License

This project is licensed under the [MIT License](LICENSE).