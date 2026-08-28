# ETA Prediction MLOps Pipeline

## System Requirements & Setup
1. Install dependencies:
   pip install -r requirements.txt
   or
   pip install pandas numpy scikit-learn fastapi uvicorn streamlit joblib requests

## How to Run the End-to-End Pipeline
1. Run Data Ingestion & Feature Engineering:
   python data_pipeline.py

2. Run Model Training & Serialization:
   python train.py

3. Launch Production FastAPI Server:
   Terminal 1: Start FastAPI Backend -
   uvicorn app:app --reload --port 8000
   Terminal 2: Start Streamlit Dashboard -
   streamlit run dashboard.py

4. Run API Unit Tests (in a second terminal):
   python test_api.py

5. Run Concept Drift Simulation & Auto-Retraining:
   python simulate_drift.py

Architecture Diagram:

ETA Prediction MLOps Pipeline - 

   Raw Data
   │
   ▼
data_pipeline.py
(Data Validation & Feature Engineering)
   │
   ▼
train.py
(Model Training)
   │
   ▼
best_model.joblib
   │
   ▼
FastAPI (app.py) <──── Streamlit Dashboard (dashboard.py)
   │
   ▼
Prediction Logs
   │
   ▼
simulate_drift.py
   │
   ▼
Drift Detection
   │
   ▼
Auto Retraining
   │
   ▼
best_model_v2.joblib

