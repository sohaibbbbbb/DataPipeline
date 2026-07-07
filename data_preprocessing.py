import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataPipelineProcessor:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.df = None
        self.model_pipeline = None

    def load_data(self):
        logging.info("Loading CSV into DataFrame...")
        self.df = pd.read_csv(self.file_path)

    def clean_commas(self):
        logging.info("Removing commas from string data...")
        self.df = self.df.replace(',', '', regex=True)

    def build_and_train_pipeline(self):
        logging.info("Building scikit-learn preprocessor and model pipeline...")
        
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        self.model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])

        logging.info("Executing fit: Preprocessing data and training Random Forest...")
        self.model_pipeline.fit(X, y)
        logging.info("Pipeline executed and model trained successfully.")
        
        return self.model_pipeline

    def run(self):
        self.load_data()
        self.clean_commas()
        return self.build_and_train_pipeline()

if __name__ == "__main__":
    processor = DataPipelineProcessor('raw_data.csv',target_column='label')
    trained_pipeline = processor.run()
    predictions = trained_pipeline.predict(new_data)
    pass