import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path=None, verbose=True):
        self.data_path = data_path
        self.verbose = verbose
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _print(self, msg):
        if self.verbose:
            print(msg)

    def load_data(self):
        """Load data from the specified path."""
        try:
            self.data = pd.read_csv(self.data_path)
            self._print(f"Data loaded successfully from {self.data_path}")
        except FileNotFoundError:
            self._print(f"File not found: {self.data_path}")
            raise

    def preprocess_data(self):
        """Preprocess the loaded data."""
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        self.data = self.data.drop(columns=['Unnamed: 32', 'id'])
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        self.X = self.data.drop('diagnosis', axis=1)
        self.y = self.data['diagnosis']

        self._print("Data preprocessing completed.")

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        if self.X is None or self.y is None:
            raise ValueError("Data is not preprocessed. Call preprocess_data() first.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        self._print(f"Data split completed. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def get_train_data(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data is not split. Call split_data() first.")

        return self.X_train, self.y_train

    def get_test_data(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data is not split. Call split_data() first.")

        return self.X_test, self.y_test

    def get_feature_names(self):
        if self.X is None:
            raise ValueError("Data is not preprocessed. Call preprocess_data() first.")

        return self.X.columns.tolist()

    def save_data(self, X_train_path, X_test_path, y_train_path, y_test_path):
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise ValueError("Data is not split. Call split_data() first.")

        pd.DataFrame(self.X_train, columns=self.get_feature_names()).to_csv(X_train_path, index=False)
        pd.DataFrame(self.X_test, columns=self.get_feature_names()).to_csv(X_test_path, index=False)
        self.y_train.to_csv(y_train_path, index=False)
        self.y_test.to_csv(y_test_path, index=False)

        print(f"Data saved successfully to {X_train_path}, {X_test_path}, {y_train_path}, and {y_test_path}")

    def load_preprocessed_data(self, X_train_path, X_test_path, y_train_path, y_test_path):
        try:
            self.X_train = np.loadtxt(X_train_path, delimiter=',', skiprows=1)
            self.X_test = np.loadtxt(X_test_path, delimiter=',', skiprows=1)
            self.y_train = pd.read_csv(y_train_path)['diagnosis'].values
            self.y_test = pd.read_csv(y_test_path)['diagnosis'].values
            self._print(f"Preprocessed data loaded successfully from {X_train_path}, {X_test_path}, {y_train_path}, and {y_test_path}")
        except FileNotFoundError:
            self._print(f"One or more files not found: {X_train_path}, {X_test_path}, {y_train_path}, {y_test_path}")
            raise

    def get_train_data(self):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data is not split. Call split_data() first.")

        return self.X_train, pd.Series(self.y_train)

    def get_test_data(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data is not split. Call split_data() first.")

        return self.X_test, pd.Series(self.y_test)
    
    def feature_names(self, X_train_path):
        X_train = pd.read_csv(X_train_path)
        feature_names = X_train.columns.tolist()
        return feature_names


if __name__ == '__main__':
    data_path = 'data/raw/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()
    feature_names = processor.get_feature_names()
    X_train_path = 'data/processed/X_train.csv'
    X_test_path = 'data/processed/X_test.csv'
    y_train_path = 'data/processed/y_train.csv'
    y_test_path = 'data/processed/y_test.csv'
    processor.save_data(X_train_path, X_test_path, y_train_path, y_test_path)
    print(f"Feature names: {feature_names}")