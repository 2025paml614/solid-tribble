"""
test_api.py

Unit testing for Week 3 (FastAPI Deployment).
This script uses FastAPI's TestClient to verify the /health and /predict endpoints,
testing correct schema handling, standard prediction inputs, and validation edge cases.
"""

import sys
import os

# Append pipeline path to sys.path
sys.path.append(r"C:\Users\devik\Downloads")

from fastapi.testclient import TestClient
from app import app

def test_health():
    """Test that the health endpoint returns 200 and correct status."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "unhealthy"]
        print("Health check endpoint test passed!")

def test_predict_success():
    """Test that a valid prediction request returns 200 and a realistic prediction."""
    payload = {
        "pickup_latitude": 40.7128,
        "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7306,
        "dropoff_longitude": -73.9352,
        "pickup_datetime": "2026-08-26 12:00:00",
        "weather": "Clear"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_eta_minutes" in data
        assert data["predicted_eta_minutes"] >= 2.0
        assert "distance_km" in data["engineered_features"]
        print("Prediction success test passed!")

def test_predict_invalid_weather():
    """Test that an invalid weather string returns a 422 Unprocessable Entity error."""
    payload = {
        "pickup_latitude": 40.7128,
        "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7306,
        "dropoff_longitude": -73.9352,
        "pickup_datetime": "2026-08-26 12:00:00",
        "weather": "Hurricanic" # Invalid weather
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        print("Invalid weather validation test passed!")

def test_predict_invalid_datetime():
    """Test that a malformed datetime string returns a 422 Unprocessable Entity error."""
    payload = {
        "pickup_latitude": 40.7128,
        "pickup_longitude": -74.0060,
        "dropoff_latitude": 40.7306,
        "dropoff_longitude": -73.9352,
        "pickup_datetime": "Aug 26, 2026", # Invalid format
        "weather": "Clear"
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        print("Invalid datetime validation test passed!")

def run_tests():
    print("Starting API unit tests...")
    test_health()
    test_predict_success()
    test_predict_invalid_weather()
    test_predict_invalid_datetime()
    print("All unit tests passed successfully!")

if __name__ == "__main__":
    run_tests()
