from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predictor import HousePricePredictor
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os


# App and Predictor Initialization
app = FastAPI(title="USA House Price API")

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

predictor = HousePricePredictor("usa_house_price_api/house_price_api_model.pkl")

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
    # get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))  
    # Default host to 0.0.0.0 to allow external access
    uvicorn.run(app, host="0.0.0.0", port=port)
