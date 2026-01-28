import pickle
import pandas as pd
import numpy as np

class HousePricePredictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            artifacts = pickle.load(f)
        self.model = artifacts["model"]
        self.encoder = artifacts["encoder"]
        self.scaler = artifacts["scaler"]

        # Final features that model expects
        self.final_features = [
            "bathrooms", "sqft_living", "floors", "view",
            "sqft_basement", "city", "statezip", "living_per_bedroom"
        ]

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the derived feature living_per_bedroom"""
        df['living_per_bedroom'] = df['sqft_living'] / df['bedrooms']
        return df

    def log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log transform skewed features"""
        log_cols = ['sqft_living', 'sqft_basement', 'living_per_bedroom']
        df[log_cols] = df[log_cols].apply(np.log1p)
        return df

    def encode_and_scale(self, df: pd.DataFrame) -> np.ndarray:
        """Encode categorical & scale numerical features"""
        df_encoded = self.encoder.transform(df)
        df_scaled = self.scaler.transform(df_encoded)
        return df_scaled

    def preprocess(self, data: dict) -> np.ndarray:
        """Complete preprocessing pipeline"""
        df_input = pd.DataFrame([data])
        df_input = self.feature_engineering(df_input)
        df_model = df_input[self.final_features].copy()
        df_model = self.log_transform(df_model)
        return self.encode_and_scale(df_model)

    def predict(self, data: dict) -> float:
        """Predict house price in USD"""
        processed = self.preprocess(data)
        prediction_log = self.model.predict(processed)
        prediction_usd = np.expm1(prediction_log)[0]
        return round(float(prediction_usd), 2)