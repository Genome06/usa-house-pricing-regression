import pytest
import pandas as pd
import numpy as np
from predictor import HousePricePredictor

# Initialize predictor once for all tests
predictor = HousePricePredictor("house_price_api_model.pkl")

# Sample input data for tests
sample_input = {
    "bathrooms": 2.5,
    "sqft_living": 2100,
    "floors": 2,
    "view": 0,
    "sqft_basement": 0,
    "city": "Seattle",
    "statezip": "WA 98103",
    "bedrooms": 3
}

def test_feature_engineering():
    """Make sure derived feature is added correctly"""
    df = pd.DataFrame([sample_input])
    df_out = predictor.feature_engineering(df.copy())
    assert "living_per_bedroom" in df_out.columns
    assert df_out.loc[0, "living_per_bedroom"] == pytest.approx(2100/3)

def test_log_transform():
    """Make sure log transform works on skewed columns"""
    df = pd.DataFrame([sample_input])
    df["living_per_bedroom"] = 2100/3
    df_out = predictor.log_transform(df.copy())
    for col in ["sqft_living", "sqft_basement", "living_per_bedroom"]:
        assert df_out[col].iloc[0] == pytest.approx(np.log1p(df[col].iloc[0]))

def test_encode_and_scale():
    """Make sure encode & scale results in a numpy array"""
    df = pd.DataFrame([sample_input])
    df["living_per_bedroom"] = 2100/3
    df_out = predictor.log_transform(df.copy())
    df_out = df_out[predictor.final_features].copy()
    arr = predictor.encode_and_scale(df_out)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 1  # hanya satu row

def test_preprocess_pipeline():
    """Make sure preprocess produces an array ready for prediction"""
    arr = predictor.preprocess(sample_input)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 1

def test_predict_output():
    """Make sure prediction returns a positive float"""
    prediction = predictor.predict(sample_input)
    assert isinstance(prediction, float)
    assert prediction > 0