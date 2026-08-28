"""
train.py

Week 2 (Module M3): Model Training, Hyperparameter Tuning, and Experiment Tracking.
This script loads the engineered feature dataset, splits it into train/test sets,
trains a Linear/Ridge Regression baseline and multiple tuned Gradient Boosting Regressors,
tracks experiments (hyperparameters, metrics) in a local JSON registry,
selects the best model based on validation RMSE, and saves it.
"""

import os
import json
import joblib
import logging
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainPipeline")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

FEATURE_DATA_PATH = os.path.join(DATA_DIR, "features_v1.csv")
EXPERIMENT_LOG_PATH = os.path.join(LOGS_DIR, "experiments.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

def load_and_split_data():
    """Loads the features data and splits it into train and test sets."""
    if not os.path.exists(FEATURE_DATA_PATH):
        raise FileNotFoundError(f"Feature dataset not found at {FEATURE_DATA_PATH}. Please run data_pipeline.py first!")
        
    df = pd.read_csv(FEATURE_DATA_PATH)
    logger.info(f"Loaded features dataset. Shape: {df.shape}")
    
    # Define feature columns and target
    feature_cols = [
        "distance_km", "hour_of_day", "day_of_week", "is_weekend",
        "weather_Clear", "weather_Rainy", "weather_Snowy"
    ]
    target_col = "actual_duration"
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Split data into train (size={len(X_train)}) and test (size={len(X_test)}) sets.")
    
    return X_train, X_test, y_train, y_test, feature_cols

def log_experiment(run_id, model_name, params, metrics):
    """Saves experiment configurations and metrics to a local JSON registry (simulating MLflow)."""
    run_entry = {
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "hyperparameters": params,
        "metrics": metrics
    }
    
    runs = []
    if os.path.exists(EXPERIMENT_LOG_PATH):
        try:
            with open(EXPERIMENT_LOG_PATH, "r") as f:
                runs = json.load(f)
        except Exception:
            logger.warning("Could not read existing experiment log. Initializing new log.")
            
    runs.append(run_entry)
    
    with open(EXPERIMENT_LOG_PATH, "w") as f:
        json.dump(runs, f, indent=4)
        
    logger.info(f"Logged experiment run '{run_id}' for {model_name}.")

def train_and_evaluate():
    X_train, X_test, y_train, y_test, feature_cols = load_and_split_data()
    
    runs_to_try = [
        # 1. Baseline Ridge Regression
        {
            "model_type": "Ridge",
            "params": {"alpha": 1.0, "random_state": 42},
            "name": "Ridge Baseline"
        },
        # 2. Gradient Boosting (Lightweight Config)
        {
            "model_type": "GradientBoosting",
            "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
            "name": "GradientBoosting - Safe Default"
        },
        # 3. Gradient Boosting (Tuned Config)
        {
            "model_type": "GradientBoosting",
            "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4, "random_state": 42},
            "name": "GradientBoosting - High Performance"
        }
    ]
    
    best_rmse = float("inf")
    best_model = None
    best_run = {}
    
    for idx, run_cfg in enumerate(runs_to_try):
        model_type = run_cfg["model_type"]
        params = run_cfg["params"]
        run_name = run_cfg["name"]
        run_id = f"RUN_{idx+1:03d}_{model_type.upper()}"
        
        logger.info(f"Starting {run_name} training...")
        start_time = time.time()
        
        # Instantiate and fit
        if model_type == "Ridge":
            model = Ridge(**params)
        elif model_type == "GradientBoosting":
            model = GradientBoostingRegressor(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Predict & Evaluate
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        metrics = {
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "test_mae": float(test_mae),
            "test_r2": float(test_r2),
            "training_time_sec": float(training_time)
        }
        
        logger.info(f"Results for {run_id}: Test RMSE = {test_rmse:.4f} | R2 = {test_r2:.4f}")
        
        # Log to registry
        log_experiment(run_id, run_name, params, metrics)
        
        # Update best model selection
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model = model
            best_run = {
                "run_id": run_id,
                "model_name": run_name,
                "hyperparameters": params,
                "metrics": metrics,
                "feature_cols": feature_cols
            }
            
    # Serialize the absolute best model
    logger.info(f"Selecting best model: {best_run['model_name']} ({best_run['run_id']}) with Test RMSE = {best_rmse:.4f}")
    
    # Save model binary
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info(f"Serialized best model saved to {BEST_MODEL_PATH}")
    
    # Save model metadata
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(best_run, f, indent=4)
    logger.info(f"Best model metadata saved to {MODEL_METADATA_PATH}")
    
    return best_run

from datetime import datetime
if __name__ == "__main__":
    train_and_evaluate()
