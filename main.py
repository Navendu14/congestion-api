from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Load model
model = joblib.load("new_model.pkl")  # Use your actual model filename

# Initialize app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class CongestionInput(BaseModel):
    is_holiday: int
    is_weekend: int
    Friday: int
    Monday: int
    Saturday: int
    Sunday: int
    Thursday: int
    Tuesday: int
    Wednesday: int
    Cloudy: int
    Rainy: int
    Sunny: int
    Windy: int
    tentotenthirty: int
    tenthirtytoeleven: int
    eleventoeleventhirty: int
    eleventhirtytotwelve: int
    twelvetotwelvethirty: int
    twelvethirtytothirteen: int
    thirteentothirteenthirty: int
    thirteenthirtytofourteen: int
    fourteentofourteenthirty: int
    fourteenthirtytofifteen: int
    fifteentofifteenthirty: int
    fifteenthirtytosixteen: int
    sixteentosixteenthirty: int
    sixteenthirtytoseventeen: int
    seventeentoseventeenthirty: int
    seventeenthirtytoeighteen: int
    eighteentoeighteenthirty: int
    eighteenthirtytonineteen: int
    nineteentonineteenthirty: int
    nineteenthirtytotwenty: int
    eighttoeightthirty: int
    eightthirtytonine: int
    ninetoninethirty: int
    ninethirtytoten: int

@app.get("/")
def home():
    return {"message": "🚦 Congestion Prediction API running!"}

@app.post("/predict")
def predict(data: CongestionInput):
    input_features = np.array([[
        data.is_holiday, data.is_weekend,
        data.Friday, data.Monday, data.Saturday, data.Sunday,
        data.Thursday, data.Tuesday, data.Wednesday,
        data.Cloudy, data.Rainy, data.Sunny, data.Windy,
        data.tentotenthirty, data.tenthirtytoeleven, data.eleventoeleventhirty,
        data.eleventhirtytotwelve, data.twelvetotwelvethirty, data.twelvethirtytothirteen,
        data.thirteentothirteenthirty, data.thirteenthirtytofourteen, data.fourteentofourteenthirty,
        data.fourteenthirtytofifteen, data.fifteentofifteenthirty, data.fifteenthirtytosixteen,
        data.sixteentosixteenthirty, data.sixteenthirtytoseventeen, data.seventeentoseventeenthirty,
        data.seventeenthirtytoeighteen, data.eighteentoeighteenthirty, data.eighteenthirtytonineteen,
        data.nineteentonineteenthirty, data.nineteenthirtytotwenty, data.eighttoeightthirty,
        data.eightthirtytonine, data.ninetoninethirty, data.ninethirtytoten
    ]])

    prediction = model.predict(input_features)
    return {"congestion_percentage": round(float(prediction[0]), 2)}
