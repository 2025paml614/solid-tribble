import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="ETA Prediction Dashboard", page_icon="🚖", layout="wide")

st.title("🚖 Real-Time ETA Prediction & MLOps Dashboard")
st.markdown("Interactive UI powered by **FastAPI** and **Gradient Boosting Regressor**.")

# Sidebar Controls
st.sidebar.header("📍 Trip Parameters")

pickup_lat = st.sidebar.number_input("Pickup Latitude", value=12.9716, format="%.4f")
pickup_lon = st.sidebar.number_input("Pickup Longitude", value=77.5946, format="%.4f")
dropoff_lat = st.sidebar.number_input("Dropoff Latitude", value=12.9352, format="%.4f")
dropoff_lon = st.sidebar.number_input("Dropoff Longitude", value=77.6245, format="%.4f")

weather = st.sidebar.selectbox("Weather Condition", ["Clear", "Rainy", "Snowy", "Foggy"])
pickup_time = st.sidebar.time_input("Pickup Time", value=datetime.now().time())
pickup_date = st.sidebar.date_input("Pickup Date", value=datetime.now().date())

combined_datetime = datetime.combine(pickup_date, pickup_time).strftime("%Y-%m-%d %H:%M:%S")

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("🗺️ Route Visualization")
    
    # Map visual using Plotly
    fig = go.Figure()

    # Add Pickup Point
    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lat=[pickup_lat],
        lon=[pickup_lon],
        marker=dict(size=14, color="green"),
        text=["Pickup Location"],
        name="Pickup"
    ))

    # Add Dropoff Point
    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lat=[dropoff_lat],
        lon=[dropoff_lon],
        marker=dict(size=14, color="red"),
        text=["Dropoff Location"],
        name="Dropoff"
    ))

    # Draw Route Line
    fig.add_trace(go.Scattermap(
        mode="lines",
        lat=[pickup_lat, dropoff_lat],
        lon=[pickup_lon, dropoff_lon],
        line=dict(width=4, color="blue"),
        name="Direct Route"
    ))

    center_lat = (pickup_lat + dropoff_lat) / 2
    center_lon = (pickup_lon + dropoff_lon) / 2

    fig.update_layout(
        map_style="open-street-map",
        map=dict(center=dict(lat=center_lat, lon=center_lon), zoom=11),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚡ Live Model Inference")
    
    if st.button("Predict ETA", type="primary"):
        payload = {
            "pickup_latitude": pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "pickup_datetime": combined_datetime,
            "weather": weather
        }

        try:
            res = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                
                st.metric(
                    label="Estimated Time of Arrival", 
                    value=f"{data['predicted_eta_minutes']:.1f} mins"
                )
                
                st.success("Prediction complete!")
                
                st.markdown("### 🔍 Model Telemetry & Features")
                st.json({
                    "Model Version": data["model_version_used"],
                    "Prediction ID": data["prediction_id"],
                    "Haversine Distance (km)": data["engineered_features"].get("haversine_distance_km", "N/A"),
                    "Is Rush Hour": data["engineered_features"].get("is_rush_hour", "N/A"),
                    "Timestamp": data["timestamp"]
                })
            else:
                st.error(f"Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error("Could not reach FastAPI backend. Make sure `uvicorn app:app --port 8000` is running!")