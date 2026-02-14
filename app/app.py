"""
Breast Cancer Prediction Application

A comprehensive Streamlit application for breast cancer prediction
with interactive visualizations and model interpretability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import DataProcessor

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #e74c3c;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-malignant {
        background-color: #ffebee;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .prediction-benign {
        background-color: #e8f5e9;
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Cache data loading functions
@st.cache_data
def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load raw data for visualization."""
    df = pd.read_csv(data_path)
    cols_to_drop = ['id']
    if 'Unnamed: 32' in df.columns:
        cols_to_drop.append('Unnamed: 32')
    return df.drop(columns=cols_to_drop, errors='ignore')


@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler."""
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler


@st.cache_data
def get_feature_names() -> list:
    """Get feature names from processed data."""
    X_train = pd.read_csv('data/processed/X_train.csv')
    return X_train.columns.tolist()


def create_gauge_chart(value: float, title: str) -> go.Figure:
    """Create a gauge chart for probability display."""
    color = "#e74c3c" if value > 0.5 else "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_feature_importance_chart(model, feature_names: list, input_values: dict) -> go.Figure:
    """Create feature importance visualization."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(10)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['importance'],
            y=importance_df['feature'],
            orientation='h',
            marker_color='#3498db',
            text=[f"{v:.3f}" for v in importance_df['importance']],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 10 Most Important Features',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=400,
            margin=dict(l=150, r=50, t=50, b=50)
        )
        
        return fig
    except Exception:
        return None


def create_input_comparison_chart(input_values: dict, data: pd.DataFrame, feature_names: list) -> go.Figure:
    """Create chart comparing input values to dataset distribution."""
    # Normalize values for comparison
    input_normalized = []
    benign_means = []
    malignant_means = []
    
    top_features = feature_names[:10]  # Show top 10 features
    
    for feature in top_features:
        feature_data = data[feature]
        input_val = input_values.get(feature, 0)
        
        # Normalize to 0-1 range
        min_val, max_val = feature_data.min(), feature_data.max()
        if max_val - min_val > 0:
            normalized = (input_val - min_val) / (max_val - min_val)
            benign_norm = (data[data['diagnosis'] == 'B'][feature].mean() - min_val) / (max_val - min_val)
            malignant_norm = (data[data['diagnosis'] == 'M'][feature].mean() - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
            benign_norm = 0.5
            malignant_norm = 0.5
        
        input_normalized.append(normalized)
        benign_means.append(benign_norm)
        malignant_means.append(malignant_norm)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=input_normalized + [input_normalized[0]],
        theta=top_features + [top_features[0]],
        fill='toself',
        name='Patient Input',
        line_color='#9b59b6'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=benign_means + [benign_means[0]],
        theta=top_features + [top_features[0]],
        fill='toself',
        name='Benign Average',
        line_color='#2ecc71',
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=malignant_means + [malignant_means[0]],
        theta=top_features + [top_features[0]],
        fill='toself',
        name='Malignant Average',
        line_color='#e74c3c',
        opacity=0.5
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Patient Values vs. Class Averages (Normalized)',
        height=500
    )
    
    return fig


def prediction_page():
    """Render the prediction page."""
    st.markdown('<p class="main-header">🏥 Breast Cancer Prediction</p>', unsafe_allow_html=True)
    
    # Load model and data
    try:
        model, scaler = load_model_and_scaler()
        feature_names = get_feature_names()
        raw_data = load_raw_data('data/raw/data.csv')
    except Exception as e:
        st.error(f"Error loading model or data: {e}")
        st.info("Please ensure models are trained by running: `python src/model_training.py`")
        return
    
    # Information box
    st.markdown("""
    <div class="info-box">
    <strong>How to use:</strong> Enter the measurements from a Fine Needle Aspiration (FNA) test below.
    The model will predict whether the tumor is likely benign or malignant based on these features.
    </div>
    """, unsafe_allow_html=True)
    
    # Predefined data (first malignant case from dataset)
    predefined_data = {
        'radius_mean': 17.99, 'texture_mean': 10.38, 'perimeter_mean': 122.80,
        'area_mean': 1001.0, 'smoothness_mean': 0.11840, 'compactness_mean': 0.27760,
        'concavity_mean': 0.30010, 'concave points_mean': 0.14710, 'symmetry_mean': 0.2419,
        'fractal_dimension_mean': 0.07871, 'radius_se': 1.0950, 'texture_se': 0.9053,
        'perimeter_se': 8.589, 'area_se': 153.40, 'smoothness_se': 0.006399,
        'compactness_se': 0.04904, 'concavity_se': 0.05373, 'concave points_se': 0.01587,
        'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193, 'radius_worst': 25.38,
        'texture_worst': 17.33, 'perimeter_worst': 184.60, 'area_worst': 2019.0,
        'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
        'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.11890
    }
    
    # Feature groups
    mean_features = [f for f in feature_names if '_mean' in f]
    se_features = [f for f in feature_names if '_se' in f]
    worst_features = [f for f in feature_names if '_worst' in f]
    
    st.markdown('<p class="sub-header">📋 Patient Measurements</p>', unsafe_allow_html=True)
    
    # Create tabs for feature groups
    tab1, tab2, tab3 = st.tabs(["📊 Mean Values", "📈 Standard Error", "⚠️ Worst Values"])
    
    form_data = {}
    
    with tab1:
        st.markdown("*Mean values of cell nucleus characteristics*")
        cols = st.columns(3)
        for i, feature in enumerate(mean_features):
            with cols[i % 3]:
                display_name = feature.replace('_mean', '').replace('_', ' ').title()
                form_data[feature] = st.number_input(
                    display_name,
                    value=predefined_data.get(feature, 0.0),
                    format="%.5f",
                    key=f"mean_{feature}"
                )
    
    with tab2:
        st.markdown("*Standard error of measurements*")
        cols = st.columns(3)
        for i, feature in enumerate(se_features):
            with cols[i % 3]:
                display_name = feature.replace('_se', '').replace('_', ' ').title() + " SE"
                form_data[feature] = st.number_input(
                    display_name,
                    value=predefined_data.get(feature, 0.0),
                    format="%.5f",
                    key=f"se_{feature}"
                )
    
    with tab3:
        st.markdown("*Largest/worst values in the sample*")
        cols = st.columns(3)
        for i, feature in enumerate(worst_features):
            with cols[i % 3]:
                display_name = feature.replace('_worst', '').replace('_', ' ').title() + " (Worst)"
                form_data[feature] = st.number_input(
                    display_name,
                    value=predefined_data.get(feature, 0.0),
                    format="%.5f",
                    key=f"worst_{feature}"
                )
    
    st.markdown("---")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔬 Analyze Sample", use_container_width=True, type="primary")
    
    if predict_button:
        # Prepare input data
        input_df = pd.DataFrame([form_data])
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)
        
        # Display prediction result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if prediction == 1:
                st.markdown("""
                <div class="prediction-malignant">
                    <h2 style="color: #c0392b; margin: 0;">⚠️ MALIGNANT</h2>
                    <p style="color: #666; margin-top: 0.5rem;">The model predicts this tumor is likely malignant.</p>
                    <p style="font-size: 0.9rem; color: #888;">Please consult with a medical professional for proper diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-benign">
                    <h2 style="color: #27ae60; margin: 0;">✅ BENIGN</h2>
                    <p style="color: #666; margin-top: 0.5rem;">The model predicts this tumor is likely benign.</p>
                    <p style="font-size: 0.9rem; color: #888;">Please consult with a medical professional for confirmation.</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Probability visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Prediction Confidence")
            fig = create_gauge_chart(prediction_proba[1], "Malignancy Probability")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Class Probabilities")
            prob_df = pd.DataFrame({
                'Class': ['Benign', 'Malignant'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            
            fig = px.bar(prob_df, x='Class', y='Probability',
                        color='Class',
                        color_discrete_map={'Benign': '#2ecc71', 'Malignant': '#e74c3c'},
                        text=[f"{p:.1%}" for p in prob_df['Probability']])
            fig.update_layout(
                yaxis_range=[0, 1],
                showlegend=False,
                height=300
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature comparison
        st.markdown("#### Patient Profile Comparison")
        radar_fig = create_input_comparison_chart(form_data, raw_data, feature_names)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Feature importance
        importance_fig = create_feature_importance_chart(model, feature_names, form_data)
        if importance_fig:
            st.markdown("#### Feature Importance")
            st.plotly_chart(importance_fig, use_container_width=True)


def eda_page():
    """Render the EDA page."""
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    try:
        data = load_raw_data('data/raw/data.csv')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Dataset overview
    st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        malignant_count = len(data[data['diagnosis'] == 'M'])
        st.metric("Malignant", f"{malignant_count} ({malignant_count/len(data)*100:.1f}%)")
    with col4:
        benign_count = len(data[data['diagnosis'] == 'B'])
        st.metric("Benign", f"{benign_count} ({benign_count/len(data)*100:.1f}%)")
    
    # Class distribution
    st.markdown('<p class="sub-header">Class Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(data, names='diagnosis', 
                     color='diagnosis',
                     color_discrete_map={'M': '#e74c3c', 'B': '#2ecc71'},
                     title='Diagnosis Distribution')
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(data, x='diagnosis', color='diagnosis',
                          color_discrete_map={'M': '#e74c3c', 'B': '#2ecc71'},
                          title='Sample Counts by Diagnosis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown('<p class="sub-header">Feature Distributions</p>', unsafe_allow_html=True)
    
    feature_cols = [c for c in data.columns if c != 'diagnosis']
    selected_feature = st.selectbox("Select Feature", feature_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(data, x=selected_feature, color='diagnosis',
                          color_discrete_map={'M': '#e74c3c', 'B': '#2ecc71'},
                          marginal='box',
                          title=f'Distribution: {selected_feature}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(data, x='diagnosis', y=selected_feature, color='diagnosis',
                    color_discrete_map={'M': '#e74c3c', 'B': '#2ecc71'},
                    title=f'Box Plot: {selected_feature}')
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown('<p class="sub-header">Feature Correlations</p>', unsafe_allow_html=True)
    
    corr_matrix = data[feature_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                   labels=dict(color="Correlation"),
                   color_continuous_scale='RdBu_r',
                   aspect='auto',
                   title='Feature Correlation Heatmap')
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Pairplot for top features
    st.markdown('<p class="sub-header">Top Features Comparison</p>', unsafe_allow_html=True)
    
    top_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'concave points_mean']
    
    fig = px.scatter_matrix(data, dimensions=top_features, color='diagnosis',
                            color_discrete_map={'M': '#e74c3c', 'B': '#2ecc71'},
                            title='Scatter Matrix: Top 5 Features')
    fig.update_layout(height=800)
    st.plotly_chart(fig, use_container_width=True)


def model_performance_page():
    """Render the model performance page."""
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    
    # Load results if available
    results_path = Path('images/training/model_results.csv')
    
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        
        st.markdown('<p class="sub-header">Model Comparison</p>', unsafe_allow_html=True)
        
        # Metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        fig = go.Figure()
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for metric, color in zip(metrics, colors):
            if metric in results_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=results_df['Model'],
                    y=results_df[metric],
                    marker_color=color,
                    text=[f"{v:.3f}" for v in results_df[metric]],
                    textposition='outside'
                ))
        
        fig.update_layout(
            barmode='group',
            title='Model Performance Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            yaxis_range=[0, 1.1],
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model highlight
        if 'F1 Score' in results_df.columns:
            best_idx = results_df['F1 Score'].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_f1 = results_df.loc[best_idx, 'F1 Score']
            
            st.success(f"🏆 **Best Model:** {best_model} with F1 Score of {best_f1:.4f}")
        
        # Detailed results table
        st.markdown('<p class="sub-header">Detailed Results</p>', unsafe_allow_html=True)
        display_cols = [c for c in ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'] 
                       if c in results_df.columns]
        st.dataframe(results_df[display_cols], use_container_width=True)
    
    # Training visualizations
    st.markdown('<p class="sub-header">Training Visualizations</p>', unsafe_allow_html=True)
    
    training_images = {
        'Model Comparison': 'images/training/model_comparison.png',
        'Combined ROC Curves': 'images/training/combined_roc_curves.png',
        'Model F1 Scores': 'images/training/model_f1_scores.png'
    }
    
    for title, path in training_images.items():
        if Path(path).exists():
            st.image(path, caption=title, use_column_width=True)


def interpretation_page():
    """Render the model interpretation page."""
    st.markdown('<p class="main-header">🔍 Model Interpretation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>About Model Interpretation:</strong> These visualizations help understand how the model 
    makes predictions and which features are most important for classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown('<p class="sub-header">Feature Importance</p>', unsafe_allow_html=True)
    
    importance_images = {
        'SHAP Beeswarm Plot': ('images/interpretation/shap_beeswarm.png', 
                               "Shows how each feature contributes to predictions. Red indicates higher feature values, blue indicates lower values."),
        'SHAP Bar Plot': ('images/interpretation/shap_bar.png',
                          "Average absolute SHAP values showing overall feature importance."),
        'Permutation Importance': ('images/interpretation/permutation_importance.png',
                                   "Shows how much model performance decreases when each feature is randomly shuffled."),
        'Feature Importance': ('images/interpretation/feature_importance.png',
                              "Direct feature importance from the model (if available).")
    }
    
    for title, (path, description) in importance_images.items():
        if Path(path).exists():
            st.markdown(f"#### {title}")
            st.markdown(f"*{description}*")
            st.image(path, use_column_width=True)
            st.markdown("---")
    
    # Partial Dependence Plot
    st.markdown('<p class="sub-header">Partial Dependence Plot</p>', unsafe_allow_html=True)
    pdp_path = 'images/interpretation/pdp_plot.png'
    if Path(pdp_path).exists():
        st.markdown("*Shows the marginal effect of each feature on predictions.*")
        st.image(pdp_path, use_column_width=True)
    
    # Correlation analysis from EDA
    st.markdown('<p class="sub-header">Feature Correlations</p>', unsafe_allow_html=True)
    
    correlation_images = {
        'Correlation Heatmap': 'images/EDA/correlation_heatmap.png',
        'Target Correlation': 'images/EDA/target_correlation.png',
        'Feature Groups Comparison': 'images/EDA/feature_groups_comparison.png'
    }
    
    for title, path in correlation_images.items():
        if Path(path).exists():
            st.markdown(f"#### {title}")
            st.image(path, use_column_width=True)


def about_page():
    """Render the about page."""
    st.markdown('<p class="main-header">ℹ️ About This Application</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Breast Cancer Prediction System
    
    This application uses machine learning to assist in breast cancer diagnosis based on 
    Fine Needle Aspiration (FNA) test results.
    
    ### Dataset
    The model is trained on the **Wisconsin Breast Cancer Dataset**, which contains features 
    computed from digitized images of FNA of breast masses. Features describe characteristics 
    of cell nuclei present in the image.
    
    ### Features
    For each cell nucleus, ten real-valued features are computed:
    - **Radius**: Mean distance from center to points on the perimeter
    - **Texture**: Standard deviation of gray-scale values
    - **Perimeter**: Perimeter of the nucleus
    - **Area**: Area of the nucleus
    - **Smoothness**: Local variation in radius lengths
    - **Compactness**: Perimeter² / Area - 1.0
    - **Concavity**: Severity of concave portions of the contour
    - **Concave Points**: Number of concave portions of the contour
    - **Symmetry**: Symmetry of the nucleus
    - **Fractal Dimension**: "Coastline approximation" - 1
    
    For each feature, three values are computed:
    - **Mean**: Average value across all cells
    - **Standard Error (SE)**: Standard error of the values
    - **Worst**: Largest value (mean of the three largest values)
    
    ### Model
    Multiple machine learning models are evaluated:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    
    The best performing model is selected based on F1 Score.
    
    ### Disclaimer
    ⚠️ **This tool is for educational purposes only and should not be used as a substitute 
    for professional medical diagnosis.** Always consult with healthcare professionals 
    for medical decisions.
    
    ---
    
    ### References
    - UCI Machine Learning Repository: Breast Cancer Wisconsin (Diagnostic) Dataset
    - [Original Paper: W.N. Street, W.H. Wolberg and O.L. Mangasarian (1993)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
    """)


def main():
    """Main application function."""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    pages = {
        "🔬 Prediction": prediction_page,
        "📊 Data Analysis": eda_page,
        "📈 Model Performance": model_performance_page,
        "🔍 Model Interpretation": interpretation_page,
        "ℹ️ About": about_page
    }
    
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Breast Cancer Prediction**  
    v2.0
    
    Built with Streamlit
    """)
    
    # Render selected page
    pages[selection]()


if __name__ == '__main__':
    main()