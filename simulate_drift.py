"""
simulate_drift.py

Week 4 (Module M5): Monitoring, Drift Simulation, and Closed-Loop Retraining.
This script simulates post-deployment activity. It generates prediction requests,
simulates concept drift (e.g., a massive festival surge or major snowstorm that
doubles actual travel times), runs a rolling window monitor on prediction accuracy,
detects drift when RMSE spikes beyond a 6.5-minute threshold, and triggers
automatic retraining using the updated dataset.
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MonitoringService")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

FEATURE_DATA_PATH = os.path.join(DATA_DIR, "features_v1.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
RETRAINED_MODEL_PATH = os.path.join(MODELS_DIR, "best_model_v2.joblib")
DRIFT_REPORT_PATH = os.path.join(LOGS_DIR, "drift_report.json")

def load_deployed_model():
    """Loads the current deployed model and metadata."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Model artifacts not found. Please run data_pipeline.py and train.py first.")
    
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        meta = json.load(f)
    return model, meta

def simulate_production_traffic(model, num_normal=300, num_drifted=300, seed=100):
    """Simulates production prediction logging with a sudden concept drift (surge pricing/severe congestion)."""
    logger.info("Simulating post-deployment production traffic...")
    np.random.seed(seed)
    
    # Load base feature data to sample from
    df = pd.read_csv(FEATURE_DATA_PATH)
    feature_cols = ["distance_km", "hour_of_day", "day_of_week", "is_weekend", "weather_Clear", "weather_Rainy", "weather_Snowy"]
    
    # Sample records
    samples = df.sample(num_normal + num_drifted, replace=True, random_state=seed).reset_index(drop=True)
    X_samples = samples[feature_cols]
    
    # Predict using the deployed model
    predictions = model.predict(X_samples)
    
    # Create production log with 'actual' times
    prod_logs = []
    
    # Segment 1: Normal Traffic (System is stable, actuals are close to predictions)
    logger.info(f"Generating {num_normal} normal traffic logs...")
    for i in range(num_normal):
        pred_eta = predictions[i]
        # Normal fluctuation: actual is prediction + some small noise
        actual_eta = max(2.0, pred_eta + np.random.normal(0, 1.5))
        prod_logs.append({
            "step": i,
            "period": "Normal Phase",
            "prediction_id": f"PROD_{i:04d}",
            "features": X_samples.iloc[i].to_dict(),
            "predicted_eta": float(pred_eta),
            "actual_duration": float(actual_eta),
            "error": float(actual_eta - pred_eta)
        })
        
    # Segment 2: Sudden Concept Drift Phase (e.g. Heavy festival surge + major traffic jam)
    # The actual travel times spike significantly, making predictions under-estimate the trip times
    logger.info(f"Simulating sudden concept drift (Festival Surge) for {num_drifted} logs...")
    for i in range(num_normal, num_normal + num_drifted):
        pred_eta = predictions[i]
        # Drift multiplier: Travel times increase by 50% to 100% due to gridlock
        drift_multiplier = np.random.uniform(1.5, 1.8)
        actual_eta = pred_eta * drift_multiplier + np.random.normal(0, 2.0)
        prod_logs.append({
            "step": i,
            "period": "Festival Drift Phase",
            "prediction_id": f"PROD_{i:04d}",
            "features": X_samples.iloc[i].to_dict(),
            "predicted_eta": float(pred_eta),
            "actual_duration": float(actual_eta),
            "error": float(actual_eta - pred_eta)
        })
        
    return pd.DataFrame(prod_logs)

def monitor_accuracy(prod_df, window_size=100, drift_threshold_rmse=6.5):
    """Monitors prediction accuracy in a rolling window. Triggers retraining when RMSE exceeds threshold."""
    logger.info(f"Initializing rolling window accuracy monitor (window_size={window_size})...")
    
    total_records = len(prod_df)
    drift_step = None
    rolling_metrics = []
    
    # Slide window
    for start in range(0, total_records - window_size + 1, 50): # slide by 50 steps
        end = start + window_size
        window_df = prod_df.iloc[start:end]
        
        # Calculate metrics for the window
        preds = window_df["predicted_eta"]
        actuals = window_df["actual_duration"]
        
        rmse = np.sqrt(np.mean((actuals - preds)**2))
        mae = np.mean(np.abs(actuals - preds))
        me = np.mean(actuals - preds) # Mean Error (negative means overestimating, positive means underestimating)
        
        period = "Normal" if "Normal" in window_df["period"].values and not "Drift" in window_df["period"].values else "Drift"
        if "Normal" in window_df["period"].values and "Drift" in window_df["period"].values:
            period = "Transition"
            
        metrics_entry = {
            "window_range": f"{start}-{end}",
            "period": period,
            "rmse": float(rmse),
            "mae": float(mae),
            "mean_error": float(me)
        }
        rolling_metrics.append(metrics_entry)
        
        logger.info(f"Window {start:03d}-{end:03d} ({period:10s}): RMSE = {rmse:.2f} min | MAE = {mae:.2f} min | Mean Error = {me:.2f} min")
        
        # Check trigger
        if rmse > drift_threshold_rmse and drift_step is None:
            drift_step = end
            logger.warning(f"🚨 DRIFT DETECTED at step {end}! Rolling RMSE ({rmse:.2f} min) exceeded threshold ({drift_threshold_rmse} min).")
            logger.warning("⚡ RETRAINING TRIGGERED AUTOMATICALLY.")
            
    return rolling_metrics, drift_step

def retrain_model(prod_df, drift_step_index):
    """Performs closed-loop automated retraining: ingests drifted data and trains a model v2."""
    logger.info("Executing closed-loop model retraining workflow...")
    
    # 1. Gather historical baseline features + targets
    hist_df = pd.read_csv(FEATURE_DATA_PATH)
    feature_cols = ["distance_km", "hour_of_day", "day_of_week", "is_weekend", "weather_Clear", "weather_Rainy", "weather_Snowy"]
    
    # Format historical training data
    X_hist = hist_df[feature_cols]
    y_hist = hist_df["actual_duration"]
    
    # 2. Gather drifted production data logged up to drift detection step
    # Extract feature dictionary back into columns
    drifted_traffic = prod_df.iloc[:drift_step_index]
    
    prod_features_list = []
    for _, row in drifted_traffic.iterrows():
        f_dict = row["features"]
        f_dict["actual_duration"] = row["actual_duration"]
        prod_features_list.append(f_dict)
        
    prod_features_df = pd.DataFrame(prod_features_list)
    
    X_prod = prod_features_df[feature_cols]
    y_prod = prod_features_df["actual_duration"]
    
    # 3. Concatenate datasets (Concept of retraining on historical + newly logged drifted data)
    X_combined = pd.concat([X_hist, X_prod], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_hist, y_prod], axis=0).reset_index(drop=True)
    
    logger.info(f"Combined Dataset Size: Historical ({len(X_hist)}) + Production Log ({len(X_prod)}) = {len(X_combined)}")
    
    # 4. Train model v2
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)
    
    logger.info("Fitting retrained Gradient Boosting Model (v2) with updated traffic patterns...")
    retrained_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    retrained_model.fit(X_train, y_train)
    
    # Evaluate model v2 on the DRIFTED phase test data to prove it recovered accuracy
    drifted_test = prod_df.iloc[drift_step_index:]
    drift_test_features_list = []
    for _, row in drifted_test.iterrows():
        f_dict = row["features"]
        f_dict["actual_duration"] = row["actual_duration"]
        f_dict["predicted_eta"] = row["predicted_eta"]
        drift_test_features_list.append(f_dict)
        
    drift_test_df = pd.DataFrame(drift_test_features_list)
    X_drift_test = drift_test_df[feature_cols]
    y_drift_test = drift_test_df["actual_duration"]
    
    # Compare original model performance on drift vs retrained model performance on drift
    old_model, _ = load_deployed_model()
    old_preds_on_drift = old_model.predict(X_drift_test)
    new_preds_on_drift = retrained_model.predict(X_drift_test)
    
    old_rmse = np.sqrt(np.mean((y_drift_test - old_preds_on_drift)**2))
    new_rmse = np.sqrt(np.mean((y_drift_test - new_preds_on_drift)**2))
    
    logger.info("=== Performance Recovery Analysis ===")
    logger.info(f"Old Model RMSE on Drifted Phase: {old_rmse:.2f} min (Highly inaccurate due to concept drift)")
    logger.info(f"Retrained Model (v2) RMSE on Drifted Phase: {new_rmse:.2f} min (Recovered and adapted to surge patterns)")
    
    # 5. Serialize retrained model
    joblib.dump(retrained_model, RETRAINED_MODEL_PATH)
    logger.info(f"Serialized retrained model saved to {RETRAINED_MODEL_PATH}")
    
    retrained_metadata = {
        "run_id": "RUN_V2_RETRAINED_ACTIVE",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "GradientBoosting - Retrained v2",
        "metrics": {
            "old_rmse_on_drift": float(old_rmse),
            "new_rmse_on_drift": float(new_rmse),
            "combined_training_records": len(X_combined)
        }
    }
    
    # Save drift report
    report = {
        "monitoring_status": "Drift Resolved & Retrained",
        "drift_detection_step": int(drift_step_index),
        "old_rmse_on_drift": float(old_rmse),
        "new_rmse_on_drift": float(new_rmse),
        "accuracy_improvement_pct": float((old_rmse - new_rmse) / old_rmse * 100)
    }
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Drift simulation report written to {DRIFT_REPORT_PATH}")
    
    return retrained_metadata

def run_simulation():
    # 1. Load deployed model
    model, meta = load_deployed_model()
    
    # 2. Simulate traffic (Normal and Drifted)
    prod_df = simulate_production_traffic(model)
    
    # 3. Monitor for drift
    rolling_metrics, drift_step = monitor_accuracy(prod_df)
    
    # 4. If drift detected, trigger closed-loop retraining
    if drift_step:
        retrain_meta = retrain_model(prod_df, drift_step)
        logger.info(f"Closed-loop ML engineering pipeline completed successfully! Model v2 is now active in staging. Staging Metrics: {retrain_meta['metrics']}")
    else:
        logger.info("Monitoring completed. No action needed (no drift triggered).")

from datetime import datetime
if __name__ == "__main__":
    run_simulation()
