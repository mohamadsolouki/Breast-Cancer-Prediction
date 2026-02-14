"""
Exploratory Data Analysis (EDA) Module for Breast Cancer Prediction

This module provides comprehensive EDA functionality including:
- Data quality assessment
- Feature distribution analysis
- Correlation analysis
- Statistical summaries
- Visualization generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class EDAAnalyzer:
    """Class for performing exploratory data analysis on breast cancer dataset."""
    
    def __init__(self, data_path: str, output_dir: str = 'images/EDA'):
        """
        Initialize EDA Analyzer.
        
        Args:
            data_path: Path to the raw data CSV file
            output_dir: Directory to save visualizations
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.feature_dist_dir = self.output_dir / 'feature_distribution'
        self.data = None
        self.feature_names = None
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dist_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load and prepare data for analysis."""
        self.data = pd.read_csv(self.data_path)
        
        # Drop unnecessary columns
        cols_to_drop = ['id']
        if 'Unnamed: 32' in self.data.columns:
            cols_to_drop.append('Unnamed: 32')
        self.data = self.data.drop(columns=cols_to_drop, errors='ignore')
        
        # Get feature names (excluding diagnosis)
        self.feature_names = [col for col in self.data.columns if col != 'diagnosis']
        
        print(f"Data loaded: {self.data.shape[0]} samples, {len(self.feature_names)} features")
        return self
    
    def get_data_summary(self) -> dict:
        """Get comprehensive data summary statistics."""
        summary = {
            'n_samples': len(self.data),
            'n_features': len(self.feature_names),
            'class_distribution': self.data['diagnosis'].value_counts().to_dict(),
            'class_balance': self.data['diagnosis'].value_counts(normalize=True).to_dict(),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicates': self.data.duplicated().sum()
        }
        return summary
    
    def print_data_quality_report(self):
        """Print a comprehensive data quality report."""
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        
        summary = self.get_data_summary()
        print(f"\n📊 Dataset Shape: {summary['n_samples']} samples × {summary['n_features']} features")
        print(f"\n📋 Class Distribution:")
        print(f"   - Malignant (M): {summary['class_distribution'].get('M', 0)} ({summary['class_balance'].get('M', 0)*100:.1f}%)")
        print(f"   - Benign (B): {summary['class_distribution'].get('B', 0)} ({summary['class_balance'].get('B', 0)*100:.1f}%)")
        print(f"\n🔍 Missing Values: {summary['missing_values']}")
        print(f"📌 Duplicate Rows: {summary['duplicates']}")
        
        # Statistical summary
        print("\n" + "-"*60)
        print("FEATURE STATISTICS")
        print("-"*60)
        stats = self.data[self.feature_names].describe()
        print(stats.T[['mean', 'std', 'min', 'max']].to_string())
        
        return summary
    
    def plot_diagnosis_distribution(self):
        """Create diagnosis class distribution visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        colors = {'M': '#e74c3c', 'B': '#2ecc71'}
        counts = self.data['diagnosis'].value_counts()
        
        axes[0].bar(counts.index, counts.values, color=[colors[x] for x in counts.index], 
                    edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Diagnosis', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Diagnosis Distribution', fontsize=14, fontweight='bold')
        
        # Add count labels
        for i, (idx, val) in enumerate(counts.items()):
            axes[0].text(i, val + 5, str(val), ha='center', fontsize=11, fontweight='bold')
        
        # Pie chart
        labels = ['Malignant (M)', 'Benign (B)']
        sizes = [counts.get('M', 0), counts.get('B', 0)]
        explode = (0.05, 0)
        
        axes[1].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    colors=['#e74c3c', '#2ecc71'], startangle=90,
                    textprops={'fontsize': 11}, shadow=True)
        axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'diagnosis_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: diagnosis_distribution.png")
    
    def plot_feature_distributions(self):
        """Create distribution plots for all features grouped by diagnosis."""
        for feature in self.feature_names:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Distribution plot with KDE
            for diagnosis, color in [('M', '#e74c3c'), ('B', '#2ecc71')]:
                data_subset = self.data[self.data['diagnosis'] == diagnosis][feature]
                label = 'Malignant' if diagnosis == 'M' else 'Benign'
                
                axes[0].hist(data_subset, bins=30, alpha=0.5, label=label, color=color, 
                            edgecolor='black', linewidth=0.5)
                
            axes[0].set_xlabel(feature, fontsize=11)
            axes[0].set_ylabel('Frequency', fontsize=11)
            axes[0].set_title(f'Distribution: {feature}', fontsize=12, fontweight='bold')
            axes[0].legend(frameon=True, fancybox=True)
            
            # Box plot
            self.data.boxplot(column=feature, by='diagnosis', ax=axes[1],
                             patch_artist=True)
            axes[1].set_xlabel('Diagnosis', fontsize=11)
            axes[1].set_ylabel(feature, fontsize=11)
            axes[1].set_title(f'Box Plot: {feature}', fontsize=12, fontweight='bold')
            plt.suptitle('')  # Remove automatic title
            
            plt.tight_layout()
            safe_name = feature.replace(' ', '_')
            plt.savefig(self.feature_dist_dir / f'dist_{safe_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Saved: {len(self.feature_names)} feature distribution plots")
    
    def plot_correlation_heatmap(self):
        """Create correlation heatmap with clustering."""
        # Calculate correlation matrix
        corr_matrix = self.data[self.feature_names].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Create clustered heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                    vmin=-1, vmax=1)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: correlation_heatmap.png")
    
    def plot_feature_correlation_with_target(self):
        """Create bar plot of feature correlations with target variable."""
        # Convert diagnosis to numeric for correlation
        data_numeric = self.data.copy()
        data_numeric['diagnosis'] = data_numeric['diagnosis'].map({'M': 1, 'B': 0})
        
        # Calculate correlations with target
        correlations = data_numeric[self.feature_names].corrwith(data_numeric['diagnosis'])
        correlations = correlations.sort_values(ascending=True)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 12))
        
        colors = ['#e74c3c' if c > 0 else '#3498db' for c in correlations.values]
        
        bars = ax.barh(correlations.index, correlations.values, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel('Correlation with Diagnosis (Malignant=1)', fontsize=12)
        ax.set_title('Feature Correlation with Target', fontsize=14, fontweight='bold')
        ax.set_xlim(-1, 1)
        
        # Add correlation values
        for bar, val in zip(bars, correlations.values):
            ax.text(val + 0.02 if val >= 0 else val - 0.08, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: target_correlation.png")
    
    def plot_pairplot_top_features(self, n_features: int = 5):
        """Create pairplot for top correlated features."""
        # Get top features by correlation with target
        data_numeric = self.data.copy()
        data_numeric['diagnosis'] = data_numeric['diagnosis'].map({'M': 1, 'B': 0})
        
        correlations = abs(data_numeric[self.feature_names].corrwith(data_numeric['diagnosis']))
        top_features = correlations.nlargest(n_features).index.tolist()
        
        # Create pairplot
        plot_data = self.data[top_features + ['diagnosis']].copy()
        
        g = sns.pairplot(plot_data, hue='diagnosis', 
                        palette={'M': '#e74c3c', 'B': '#2ecc71'},
                        diag_kind='kde', 
                        plot_kws={'alpha': 0.6, 's': 30},
                        diag_kws={'alpha': 0.6})
        
        g.fig.suptitle(f'Pairplot: Top {n_features} Features by Correlation', 
                       fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pairplot_top_features.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: pairplot_top_features.png (top {n_features} features)")
    
    def plot_feature_groups_summary(self):
        """Create summary visualization for feature groups (mean, se, worst)."""
        # Group features
        mean_features = [f for f in self.feature_names if '_mean' in f]
        se_features = [f for f in self.feature_names if '_se' in f]
        worst_features = [f for f in self.feature_names if '_worst' in f]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        groups = [
            ('Mean Features', mean_features, axes[0]),
            ('SE Features', se_features, axes[1]),
            ('Worst Features', worst_features, axes[2])
        ]
        
        for title, features, ax in groups:
            if features:
                # Calculate mean values for each diagnosis
                data_malignant = self.data[self.data['diagnosis'] == 'M'][features].mean()
                data_benign = self.data[self.data['diagnosis'] == 'B'][features].mean()
                
                # Normalize for comparison
                data_malignant_norm = (data_malignant - data_malignant.min()) / (data_malignant.max() - data_malignant.min())
                data_benign_norm = (data_benign - data_benign.min()) / (data_benign.max() - data_benign.min())
                
                x = np.arange(len(features))
                width = 0.35
                
                short_names = [f.replace('_mean', '').replace('_se', '').replace('_worst', '') for f in features]
                
                ax.bar(x - width/2, data_malignant_norm, width, label='Malignant', color='#e74c3c', alpha=0.8)
                ax.bar(x + width/2, data_benign_norm, width, label='Benign', color='#2ecc71', alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.legend(fontsize=9)
                ax.set_ylabel('Normalized Mean Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_groups_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: feature_groups_comparison.png")
    
    def identify_highly_correlated_features(self, threshold: float = 0.9) -> pd.DataFrame:
        """Identify highly correlated feature pairs."""
        corr_matrix = self.data[self.feature_names].corr().abs()
        
        # Get upper triangle
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs above threshold
        high_corr_pairs = []
        for column in upper_tri.columns:
            for index, value in upper_tri[column].items():
                if value > threshold:
                    high_corr_pairs.append({
                        'Feature 1': index,
                        'Feature 2': column,
                        'Correlation': value
                    })
        
        df = pd.DataFrame(high_corr_pairs)
        if not df.empty:
            df = df.sort_values('Correlation', ascending=False)
        
        print(f"\n📊 Found {len(df)} highly correlated feature pairs (|r| > {threshold})")
        return df
    
    def run_full_analysis(self):
        """Run complete EDA analysis and generate all visualizations."""
        print("\n" + "="*60)
        print("RUNNING FULL EDA ANALYSIS")
        print("="*60)
        
        self.load_data()
        self.print_data_quality_report()
        
        print("\n" + "-"*60)
        print("GENERATING VISUALIZATIONS")
        print("-"*60)
        
        self.plot_diagnosis_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_heatmap()
        self.plot_feature_correlation_with_target()
        self.plot_pairplot_top_features()
        self.plot_feature_groups_summary()
        
        print("\n" + "-"*60)
        print("HIGHLY CORRELATED FEATURES")
        print("-"*60)
        high_corr = self.identify_highly_correlated_features()
        if not high_corr.empty:
            print(high_corr.to_string(index=False))
        
        print("\n✅ EDA Analysis Complete!")
        print(f"📁 Visualizations saved to: {self.output_dir}")


if __name__ == '__main__':
    analyzer = EDAAnalyzer('data/raw/data.csv')
    analyzer.run_full_analysis()
