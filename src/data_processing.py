import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_registry import (
    FEATURE_COLUMNS, TARGET_COLUMN, LABEL_MAP,
    apply_label_map, encode_operation_mode, REQUIRED_RAW_COLUMNS
)

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None

        os.makedirs(self.output_path,exist_ok=True)
        logger.info("Data Processing initalized...")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data loaded sucesfully...")
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data",e)
        
    def preprocess(self):
        try:
            self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"], errors="coerce")
            self.df["Year"]  = self.df["Timestamp"].dt.year
            self.df["Month"] = self.df["Timestamp"].dt.month
            self.df["Day"]   = self.df["Timestamp"].dt.day
            self.df["Hour"]  = self.df["Timestamp"].dt.hour

            drop_cols = [c for c in ["Timestamp", "Machine_ID"] if c in self.df.columns]
            self.df.drop(columns=drop_cols, inplace=True)

            # Fixed encoding from feature_registry — deterministic across all contexts
            self.df["Operation_Mode"] = self.df["Operation_Mode"].apply(encode_operation_mode)
            self.df[TARGET_COLUMN]    = apply_label_map(self.df[TARGET_COLUMN])

            logger.info("All basic data preprocessing done..")

        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            raise CustomException("Failed to preprocess data", e)
        
    def split_and_scale_and_save(self):
        try:
            self.features = FEATURE_COLUMNS   # imported from feature_registry

            X = self.df[self.features]
            y = self.df[TARGET_COLUMN]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train , X_test , y_train , y_test = train_test_split(X_scaled,y, test_size=0.2 , random_state=42 , stratify=y)

            joblib.dump(X_train , os.path.join(self.output_path , "X_train.pkl"))
            joblib.dump(X_test , os.path.join(self.output_path , "X_test.pkl"))
            joblib.dump(y_train , os.path.join(self.output_path , "y_train.pkl"))
            joblib.dump(y_test , os.path.join(self.output_path , "y_test.pkl"))

            joblib.dump(scaler , os.path.join(self.output_path , "scaler.pkl"))
            logger.info("All things saved sucesfully for Data processing..")

        except Exception as e:
            logger.error(f"Error while split scale and save data {e}")
            raise CustomException("Failed to spli sacle and save data",e)
        
    def run(self):
        self.load_data()
        self.preprocess()
        self.split_and_scale_and_save()

if __name__=="__main__":
    processor = DataProcessing("artifacts/raw/data.csv" , "artifacts/processed")
    processor.run()


        
            

