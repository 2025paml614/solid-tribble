"""
data_pipeline.py

Week 1 (Module M2): Data Ingestion, Validation, and Feature Engineering Pipeline.
This script generates a synthetic raw delivery/ride dataset, performs data validation,
engineers predictive features (Haversine distance, time components, weather encoding),
and versions the output.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataPipeline")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_trips_v1.csv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "clean_trips_v1.csv")
FEATURE_DATA_PATH = os.path.join(DATA_DIR, "features_v1.csv")
PIPELINE_METRICS_PATH = os.path.join(DATA_DIR, "pipeline_metrics_v1.json")

def generate_raw_synthetic_data(num_records=5000, seed=42):
    """Generates synthetic historical ride-hailing data with realistic features and dirty rows."""
    logger.info(f"Generating {num_records} synthetic trip records...")
    np.random.seed(seed)
    
    # Generate realistic base coordinates around NYC (lat: 40.7, lon: -74.0)
    pickup_lat = np.random.normal(40.7128, 0.05, num_records)
    pickup_lon = np.random.normal(-74.0060, 0.05, num_records)
    
    # Delivery distance in degrees
    delta_lat = np.random.normal(0.02, 0.03, num_records)
    delta_lon = np.random.normal(0.02, 0.03, num_records)
    
    dropoff_lat = pickup_lat + delta_lat
    dropoff_lon = pickup_lon + delta_lon
    
    # Base timestamps (spread over August 2026)
    start_date = datetime(2026, 8, 1)
    pickup_times = [start_date + timedelta(
        days=int(np.random.randint(0, 25)),
        hours=int(np.random.randint(0, 24)),
        minutes=int(np.random.randint(0, 60)),
        seconds=int(np.random.randint(0, 60))
    ) for _ in range(num_records)]
    
    # Weather conditions
    weather_options = ["Clear", "Rainy", "Snowy"]
    weather = np.random.choice(weather_options, num_records, p=[0.7, 0.2, 0.1])
    
    # Calculate travel times with realistic relationships
    # Simple Haversine calculation for base duration
    lat1, lon1, lat2, lon2 = np.radians(pickup_lat), np.radians(pickup_lon), np.radians(dropoff_lat), np.radians(dropoff_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances_km = 6371 * 2 * np.arcsin(np.sqrt(a))
    
    durations_min = []
    for idx, dist in enumerate(distances_km):
        p_time = pickup_times[idx]
        hr = p_time.hour
        w = weather[idx]
        
        # Base speed ~ 30 km/h (2 mins per km)
        base_time = dist * 2.0 + 5.0 # 5 min fixed overhead
        
        # Traffic multiplier based on hour of day
        traffic_mult = 1.0
        if 8 <= hr <= 10:   # Morning rush
            traffic_mult = 1.6
        elif 17 <= hr <= 20: # Evening rush
            traffic_mult = 1.8
        elif 22 <= hr or hr <= 5: # Night
            traffic_mult = 0.8
            
        # Weather multiplier
        weather_mult = 1.0
        if w == "Rainy":
            weather_mult = 1.3
        elif w == "Snowy":
            weather_mult = 1.7
            
        duration = base_time * traffic_mult * weather_mult + np.random.normal(0, 2.0)
        durations_min.append(max(2.0, duration)) # floor at 2 minutes
        
    # Construct DataFrame
    df = pd.DataFrame({
        "trip_id": [f"TRIP_{i:05d}" for i in range(num_records)],
        "pickup_datetime": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in pickup_times],
        "pickup_latitude": pickup_lat,
        "pickup_longitude": pickup_lon,
        "dropoff_latitude": dropoff_lat,
        "dropoff_longitude": dropoff_lon,
        "weather": weather,
        "actual_duration": durations_min
    })
    
    # --- Inject Dirty Data for Validation Demonstration ---
    logger.info("Injecting dirty data for validation testing...")
    
    # 1. Missing GPS coordinates (1.5%)
    null_gps_indices = np.random.choice(num_records, int(num_records * 0.015), replace=False)
    df.loc[null_gps_indices, "pickup_latitude"] = np.nan
    df.loc[null_gps_indices, "pickup_longitude"] = np.nan
    
    # 2. Invalid timestamps (1.0% - dropoff before pickup, or garbage text)
    invalid_time_indices = np.random.choice(num_records, int(num_records * 0.01), replace=False)
    for idx in invalid_time_indices[:len(invalid_time_indices)//2]:
        df.loc[idx, "pickup_datetime"] = "GARBAGE_TIMESTAMP_TEXT"
    
    # 3. Impossible values (1.0% - negative or extreme actual duration)
    invalid_dur_indices = np.random.choice(num_records, int(num_records * 0.01), replace=False)
    df.loc[invalid_dur_indices, "actual_duration"] = -99.0
    
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Raw synthetic data saved to {RAW_DATA_PATH}")
    return df

def validate_schema(df):
    """Validates the schema, drops invalid rows, and logs data quality issues."""
    logger.info("Starting schema and data validation...")
    initial_rows = len(df)
    
    # 1. Validate Trip ID uniqueness
    duplicate_ids = df["trip_id"].duplicated().sum()
    if duplicate_ids > 0:
        logger.warning(f"Found {duplicate_ids} duplicate Trip IDs. Dropping duplicates.")
        df = df.drop_duplicates(subset=["trip_id"])
        
    # 2. Validate GPS coordinates (Must not be null and must fall within NYC boundaries roughly)
    # Valid ranges: Latitude [39.0, 42.0], Longitude [-75.0, -71.0]
    gps_null_mask = df["pickup_latitude"].isna() | df["pickup_longitude"].isna() | df["dropoff_latitude"].isna() | df["dropoff_longitude"].isna()
    gps_out_of_bounds = (
        (df["pickup_latitude"] < 39.0) | (df["pickup_latitude"] > 42.0) |
        (df["pickup_longitude"] < -75.0) | (df["pickup_longitude"] > -71.0) |
        (df["dropoff_latitude"] < 39.0) | (df["dropoff_latitude"] > 42.0) |
        (df["dropoff_longitude"] < -75.0) | (df["dropoff_longitude"] > -71.0)
    )
    invalid_gps_mask = gps_null_mask | gps_out_of_bounds
    num_invalid_gps = invalid_gps_mask.sum()
    logger.info(f"Invalid GPS coordinates found: {num_invalid_gps} rows.")
    
    # 3. Validate Timestamps
    # Attempt to parse pickup_datetime
    parsed_timestamps = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    invalid_time_mask = parsed_timestamps.isna()
    num_invalid_time = invalid_time_mask.sum()
    logger.info(f"Malformed timestamps found: {num_invalid_time} rows.")
    
    # 4. Validate Targets (duration must be positive and less than 3 hours / 180 mins)
    invalid_duration_mask = (df["actual_duration"] <= 0) | (df["actual_duration"] > 180.0) | df["actual_duration"].isna()
    num_invalid_dur = invalid_duration_mask.sum()
    logger.info(f"Invalid actual durations found: {num_invalid_dur} rows.")
    
    # Combine masks and drop bad rows
    total_invalid_mask = invalid_gps_mask | invalid_time_mask | invalid_duration_mask
    valid_df = df[~total_invalid_mask].copy()
    
    # Cast validated datetime columns
    valid_df["pickup_datetime"] = pd.to_datetime(valid_df["pickup_datetime"])
    
    final_rows = len(valid_df)
    logger.info(f"Validation complete. Rows ingested: {initial_rows} | Rows kept: {final_rows} | Rows dropped: {initial_rows - final_rows}")
    
    # Save a metrics dict
    validation_metrics = {
        "initial_rows": int(initial_rows),
        "duplicate_ids_dropped": int(duplicate_ids),
        "invalid_gps_dropped": int(num_invalid_gps),
        "invalid_time_dropped": int(num_invalid_time),
        "invalid_duration_dropped": int(num_invalid_dur),
        "final_valid_rows": int(final_rows),
        "success_rate": float(final_rows / initial_rows)
    }
    
    valid_df.to_csv(CLEAN_DATA_PATH, index=False)
    logger.info(f"Cleaned validated data saved to {CLEAN_DATA_PATH}")
    
    return valid_df, validation_metrics

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in kilometers."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of Earth in kilometers
    return c * r

def engineer_features(df):
    """Performs feature engineering: Extracts time indicators, distance, and maps categorical weather."""
    logger.info("Engineering features...")
    features_df = df.copy()
    
    # 1. Distance Calculation (Haversine)
    features_df["distance_km"] = calculate_haversine_distance(
        features_df["pickup_latitude"], features_df["pickup_longitude"],
        features_df["dropoff_latitude"], features_df["dropoff_longitude"]
    )
    
    # 2. Time-based features
    features_df["hour_of_day"] = features_df["pickup_datetime"].dt.hour
    features_df["day_of_week"] = features_df["pickup_datetime"].dt.weekday # Monday=0, Sunday=6
    features_df["is_weekend"] = (features_df["day_of_week"] >= 5).astype(int)
    
    # 3. Categorical Weather encoding (One-Hot Encoding to avoid ordinal bias)
    # For simplicity, we also keep the raw weather column but create dummy vars
    weather_dummies = pd.get_dummies(features_df["weather"], prefix="weather", dtype=int)
    
    # Ensure all possible categories are represented
    for col in ["weather_Clear", "weather_Rainy", "weather_Snowy"]:
        if col not in weather_dummies.columns:
            weather_dummies[col] = 0
            
    features_df = pd.concat([features_df, weather_dummies], axis=1)
    
    # Drop raw text / location coordinates to create clean training matrix
    # Keep trip_id and pickup_datetime for traceability/versioning
    features_df.to_csv(FEATURE_DATA_PATH, index=False)
    logger.info(f"Feature dataset engineered and versioned as {FEATURE_DATA_PATH}")
    
    return features_df

def run_pipeline():
    # 1. Ingest / Generate raw data
    raw_df = generate_raw_synthetic_data()
    
    # 2. Validate
    valid_df, metrics = validate_schema(raw_df)
    
    # 3. Feature Engineer
    features_df = engineer_features(valid_df)
    
    # Save pipeline metrics
    metrics["engineered_features_count"] = int(features_df.shape[1])
    metrics["final_feature_columns"] = list(features_df.columns)
    
    with open(PIPELINE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Data Pipeline run completed successfully! Metrics logged to {PIPELINE_METRICS_PATH}")
    return features_df

if __name__ == "__main__":
    run_pipeline()
