"""
app.py

Week 3 (Module M4): Model Packaging and Deployment.
This script sets up a production-ready REST API using FastAPI.
It loads the serialized best model, validates incoming trip payloads via Pydantic,
engineers predictive features on-the-fly, generates ETA predictions,
logs the prediction inputs/outputs to a persistent log file, and returns the response.
"""

import os
import json
import joblib
import logging
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, "prediction_logs.jsonl")

# Initialize FastAPI app
app = FastAPI(
    title="Delivery & Ride ETA Prediction Service",
    description="A service that predicts the delivery or ride ETA based on spatial, temporal, and weather inputs.",
    version="1.0.0"
)

# Global variables for model state
model = None
model_metadata = None

@app.on_event("startup")
def load_service_artifacts():
    """Loads the serialized model and metadata on startup."""
    global model, model_metadata
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}. API will start, but prediction requests will fail.")
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Successfully loaded model binary.")
    except Exception as e:
        logger.critical(f"Failed to load model binary: {e}")
        
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded model metadata. Best model: {model_metadata.get('model_name')}")
        except Exception as e:
            logger.error(f"Failed to load model metadata: {e}")

# --- Pydantic Schemas for Input Validation ---

class TripRequest(BaseModel):
    pickup_latitude: float = Field(..., ge=-90.0, le=90.0, description="Pickup latitude", example=40.7128)
    pickup_longitude: float = Field(..., ge=-180.0, le=180.0, description="Pickup longitude", example=-74.0060)
    dropoff_latitude: float = Field(..., ge=-90.0, le=90.0, description="Dropoff latitude", example=40.7306)
    dropoff_longitude: float = Field(..., ge=-180.0, le=180.0, description="Dropoff longitude", example=-73.9352)
    pickup_datetime: str = Field(..., description="Timestamp of pickup (format: YYYY-MM-DD HH:MM:SS)", example="2026-08-26 12:00:00")
    weather: str = Field(..., description="Weather status: Clear, Rainy, or Snowy", example="Clear")

    @field_validator("weather")
    @classmethod
    def validate_weather(cls, v: str) -> str:
        valid_options = ["Clear", "Rainy", "Snowy"]
        if v.title() not in valid_options:
            raise ValueError(f"Weather must be one of {valid_options}")
        return v.title()

    @field_validator("pickup_datetime")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("pickup_datetime must be in YYYY-MM-DD HH:MM:SS format")
        return v

class ETAPredictionResponse(BaseModel):
    prediction_id: str
    predicted_eta_minutes: float
    engineered_features: dict
    model_version_used: str
    timestamp: str

# --- Helper Functions ---

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Earth radius
    return float(c * r)

def log_prediction_to_file(payload: dict):
    """Logs prediction requests and responses to a local JSONL file for drift auditing."""
    try:
        with open(PREDICTION_LOG_PATH, "a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception as e:
        logger.error(f"Failed to write prediction log: {e}")

# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health check endpoint."""
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded", "timestamp": datetime.now().isoformat()}
    return {
        "status": "healthy",
        "model_version": model_metadata.get("run_id", "Unknown"),
        "model_type": model_metadata.get("model_name", "Unknown"),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=ETAPredictionResponse, status_code=status.HTTP_200_OK)
def predict_eta(request: TripRequest):
    """Performs feature extraction and prediction for a single trip request."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized or currently unavailable."
        )
        
    try:
        # 1. On-the-fly Feature Extraction
        dist_km = calculate_haversine_distance(
            request.pickup_latitude, request.pickup_longitude,
            request.dropoff_latitude, request.dropoff_longitude
        )
        
        # We reject ridiculous distances
        if dist_km > 200.0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Extracted distance {dist_km:.2f} km exceeds maximum service radius of 200 km."
            )
            
        dt_obj = datetime.strptime(request.pickup_datetime, "%Y-%m-%d %H:%M:%S")
        hour_of_day = dt_obj.hour
        day_of_week = dt_obj.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Weather Dummy variable encoding
        weather_clear = 1 if request.weather == "Clear" else 0
        weather_rainy = 1 if request.weather == "Rainy" else 0
        weather_snowy = 1 if request.weather == "Snowy" else 0
        
        # Assemble feature array in the exact order model expects:
        # ["distance_km", "hour_of_day", "day_of_week", "is_weekend", "weather_Clear", "weather_Rainy", "weather_Snowy"]
        feature_vector = np.array([[
            dist_km, hour_of_day, day_of_week, is_weekend,
            weather_clear, weather_rainy, weather_snowy
        ]])
        
        # 2. Inference
        predicted_eta = float(model.predict(feature_vector)[0])
        # Ensure predictions make physical sense (min floor of 2 minutes)
        predicted_eta = max(2.0, predicted_eta)
        
        # Create prediction metadata
        prediction_id = f"PRED_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        engineered_features = {
            "distance_km": dist_km,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "weather_Clear": weather_clear,
            "weather_Rainy": weather_rainy,
            "weather_Snowy": weather_snowy
        }
        
        response_data = {
            "prediction_id": prediction_id,
            "predicted_eta_minutes": round(predicted_eta, 2),
            "engineered_features": engineered_features,
            "model_version_used": model_metadata.get("run_id", "v1"),
            "timestamp": timestamp_str
        }
        
        # 3. Log request & response for post-deployment drift tracking (Week 4 requirement)
        log_payload = {
            "prediction_id": prediction_id,
            "request": request.dict(),
            "response": response_data,
            "logged_at": timestamp_str
        }
        log_prediction_to_file(log_payload)
        
        return response_data
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Inference pipeline exception: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure: {str(e)}"
        )
