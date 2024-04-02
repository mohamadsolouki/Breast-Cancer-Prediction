import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path}")
        except FileNotFoundError:
            print(f"File not found: {self.data_path}")
            raise

    def preprocess_data(self):
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        # Drop unnecessary columns
        columns_to_drop = ['Unnamed: 32', 'id']
        self.data = self.data.drop(columns=columns_to_drop)

        # Encode the target variable
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        # Separate features and target
        self.X = self.data.drop('diagnosis', axis=1)
        self.y = self.data['diagnosis']

        # Scale the features
        self.X_scaled = self.scaler.fit_transform(self.X)

        print("Data preprocessing completed.")

    def split_data(self, test_size=0.2, random_state=42):
        if self.X is None or self.y is None:
            raise ValueError("Data is not preprocessed. Call preprocess_data() first.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print(f"Data split completed. Train size: {len(self.X_train)}, Test size: {len(self.X_test)}")

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

    def save_data(self, path):
        if self.data is None:
            raise ValueError("Data is not loaded. Call load_data() first.")

        self.data.to_csv(path, index=False)
        print(f"Data saved successfully to {path}")

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)
        print(f"Scaler saved successfully to {path}")

if __name__ == '__main__':
    data_path = 'data/raw/data.csv'
    processor = DataProcessor(data_path)
    processor.load_data()
    processor.preprocess_data()
    processor.split_data()
    X_train, y_train = processor.get_train_data()
    X_test, y_test = processor.get_test_data()
    feature_names = processor.get_feature_names()
    save_path = 'data/processed/data.csv'
    processor.save_data(save_path)
    scaler_path = 'models/scaler.pkl'
    processor.save_scaler(scaler_path)