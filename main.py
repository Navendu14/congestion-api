from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load the model
model = joblib.load("congestion_model.pkl")

# Initialize app
app = FastAPI()

# Enable CORS (for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production: use specific origins like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input schema using Pydantic
class CongestionInput(BaseModel):
    holiday: int
    weekend: int
    Friday: int
    Monday: int
    Saturday: int
    Sunday: int
    Thursday: int
    Tuesday: int
    Wednesday: int
    cloudy: int
    rainy: int
    sunny: int
    windy: int


@app.get("/")
def home():
    return {"message": "Congestion Prediction API running!"}


@app.post("/predict")
def predict(data: CongestionInput):
    # Convert input to array format
    input_features = np.array([[
        data.holiday, data.weekend, data.Friday, data.Monday, data.Saturday, data.Sunday,
        data.Thursday, data.Tuesday, data.Wednesday, data.cloudy, data.rainy, data.sunny, data.windy
    ]])

    # Predict
    prediction = model.predict(input_features)
    return {"congestion_percentage": float(prediction[0])}
