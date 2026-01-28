from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import HousePricePredictor
import uvicorn

# App and Predictor Initialization
app = FastAPI(title="USA House Price API")
predictor = HousePricePredictor("usa-house-price-api/house_price_api_model.pkl")

# Data Model for Input Validation
class HouseData(BaseModel):
    bathrooms: float
    sqft_living: float
    floors: float
    view: int
    sqft_basement: float
    city: str
    statezip: str
    bedrooms: int

@app.post("/predict")
async def predict(data: HouseData):
    try:
        # FastAPI automatically converts input to dict
        result = predictor.predict(data.dict())
        return {
            "status": "success",
            "prediction_usd": result,
            "model_r2_score": 0.7547 # Final model R² score
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)