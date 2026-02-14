"""
Data Preprocessing Module for Breast Cancer Prediction

This module provides comprehensive data preprocessing including:
- Data loading and validation
- Feature engineering
- Train/test splitting with stratification
- Data scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from pathlib import Path
import joblib
from typing import Tuple, List, Optional


class DataProcessor:
    """Data processor for breast cancer prediction dataset."""
    
    def __init__(self, data_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to the raw data CSV file
            verbose: Whether to print progress messages
        """
        self.data_path = data_path
        self.verbose = verbose
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_list = None
        self.scaler = None
        
    def _print(self, msg: str):
        """Print message if verbose is enabled."""
        if self.verbose:
            print(msg)

    def load_data(self) -> 'DataProcessor':
        """Load data from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            self._print(f"Data loaded successfully from {self.data_path}")
        except FileNotFoundError:
            self._print(f"File not found: {self.data_path}")
            raise
        return self

    def preprocess_data(self) -> 'DataProcessor':
        """Preprocess the loaded data."""
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Drop unnecessary columns
        cols_to_drop = ['id']
        if 'Unnamed: 32' in self.data.columns:
            cols_to_drop.append('Unnamed: 32')
        self.data = self.data.drop(columns=cols_to_drop, errors='ignore')
        
        # Encode target variable
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        self.X = self.data.drop('diagnosis', axis=1)
        self.y = self.data['diagnosis']
        self.feature_names_list = self.X.columns.tolist()

        self._print("Data preprocessing completed.")
        return self

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> 'DataProcessor':
        """Split the data into training and testing sets with stratification."""
        if self.X is None or self.y is None:
            raise ValueError("Data is not preprocessed. Call preprocess_data() first.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.y
        )

        self._print(f"Data split completed. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")
        return self
    
    def fit_scaler(self) -> 'DataProcessor':
        """Fit StandardScaler on training data."""
        if self.X_train is None:
            raise ValueError("Data is not split. Call split_data() first.")
            
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        self._print("Scaler fitted on training data.")
        return self
    
    def get_scaled_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scaled training and test data."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
            
        X_train_scaled = self.scaler.transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        return X_train_scaled, X_test_scaled

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training features and labels."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data is not split. Call split_data() first.")
        
        y_train = self.y_train if isinstance(self.y_train, pd.Series) else pd.Series(self.y_train)
        return self.X_train, y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test features and labels."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data is not split. Call split_data() first.")
        
        y_test = self.y_test if isinstance(self.y_test, pd.Series) else pd.Series(self.y_test)
        return self.X_test, y_test

    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if self.feature_names_list is not None:
            return self.feature_names_list
        if self.X is not None:
            return self.X.columns.tolist()
        raise ValueError("Data is not preprocessed. Call preprocess_data() first.")
    
    def get_feature_importance_mi(self, k: int = 10) -> pd.DataFrame:
        """Calculate mutual information based feature importance."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data is not split. Call split_data() first.")
            
        selector = SelectKBest(mutual_info_classif, k='all')
        selector.fit(self.X_train, self.y_train)
        
        importance_df = pd.DataFrame({
            'feature': self.get_feature_names(),
            'importance': selector.scores_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(k)

    def save_data(self, X_train_path: str, X_test_path: str, 
                  y_train_path: str, y_test_path: str):
        """Save processed data to disk."""
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data is not split. Call split_data() first.")

        # Create directories if needed
        for path in [X_train_path, X_test_path, y_train_path, y_test_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        feature_names = self.get_feature_names()
        pd.DataFrame(self.X_train, columns=feature_names).to_csv(X_train_path, index=False)
        pd.DataFrame(self.X_test, columns=feature_names).to_csv(X_test_path, index=False)
        
        y_train_df = self.y_train if isinstance(self.y_train, pd.Series) else pd.Series(self.y_train, name='diagnosis')
        y_test_df = self.y_test if isinstance(self.y_test, pd.Series) else pd.Series(self.y_test, name='diagnosis')
        y_train_df.to_frame('diagnosis').to_csv(y_train_path, index=False)
        y_test_df.to_frame('diagnosis').to_csv(y_test_path, index=False)

        self._print(f"Data saved to {X_train_path}, {X_test_path}, {y_train_path}, {y_test_path}")

    def load_preprocessed_data(self, X_train_path: str, X_test_path: str, 
                                y_train_path: str, y_test_path: str):
        """Load preprocessed data from disk."""
        try:
            self.X_train = pd.read_csv(X_train_path)
            self.X_test = pd.read_csv(X_test_path)
            self.y_train = pd.read_csv(y_train_path)['diagnosis'].values
            self.y_test = pd.read_csv(y_test_path)['diagnosis'].values
            self.feature_names_list = self.X_train.columns.tolist()
            self._print(f"Preprocessed data loaded successfully")
        except FileNotFoundError as e:
            self._print(f"Error loading data: {e}")
            raise
    
    def feature_names(self, X_train_path: str) -> List[str]:
        """Get feature names from a CSV file."""
        df = pd.read_csv(X_train_path)
        return df.columns.tolist()


if __name__ == '__main__':
    # Process and save data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    
    processor = DataProcessor('data/raw/data.csv')
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()
    processor.fit_scaler()
    
    # Save scaler
    joblib.dump(processor.scaler, 'models/scaler.pkl')
    print("Scaler saved to models/scaler.pkl")
    
    # Save data
    processor.save_data(
        'data/processed/X_train.csv',
        'data/processed/X_test.csv',
        'data/processed/y_train.csv',
        'data/processed/y_test.csv'
    )
    
    # Print feature info
    print(f"\nFeature names: {processor.get_feature_names()}")
    print("\nTop 10 Features by Mutual Information:")
    print(processor.get_feature_importance_mi(10).to_string(index=False))