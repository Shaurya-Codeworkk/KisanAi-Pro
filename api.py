# ------------------- API Merged Code -------------------
# To run this API:
# 1. Install necessary packages: pip install fastapi uvicorn python-multipart joblib pandas numpy requests groq jinja2
# 2. Run the server from your terminal: uvicorn api:app --reload

import joblib
import pandas as pd
import numpy as np
import requests
import json
from groq import Groq
from datetime import datetime
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Tuple
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app import *

# --- Pydantic Models for Request & Response ---
class AgentInput(BaseModel):
    prompt: str = Field(..., example="Suggest crops for black soil in Pune, MH")

class ManualInput(BaseModel):
    city_or_state: str = Field(..., example="Pune, MH")
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    temperature: float = Field(..., example=24.5)
    humidity: float = Field(..., example=82.1)
    ph: float = Field(..., example=6.5)
    rainfall: float = Field(..., example=202.9)

class MiniChatInput(BaseModel):
    query: str = Field(..., example="tell me about wheat")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Crop Recommendation API",
    description="An intelligent API for crop recommendations using ML and AI agents.",
    version="1.3.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Template Engine Config ---
templates = Jinja2Templates(directory="templates")

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # set in Render
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # set in Render
USE_GROQ = True
CACHE_FILE = "groq_cache.json"

# --- Caching ---
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            groq_cache = json.load(f)
    else:
        groq_cache = {}
except Exception as e:
    print(f"Warning: Could not load cache file. A new one will be created. Error: {e}")
    groq_cache = {}

print("Libraries imported and app configured.")

# --- Load ML Models ---
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully loaded! ✅")
except FileNotFoundError as e:
    print(f"❌ CRITICAL ERROR: Model file not found: {e}.")
    model, scaler, encoder = None, None, None
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load ML models: {e}")
    model, scaler, encoder = None, None, None

# --- State Abbreviations Mapping ---
state_map = { "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana", "ka": "Karnataka", "tn": "Tamil Nadu" }

# (⚡ Your helper functions, AI logic, and endpoints remain SAME ⚡)
# ----------------------------------------------------------
# --- API Endpoints ---

# ✅ Root endpoint now serves index.html
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/agent", tags=["Predictions"])
def predict_with_agent(data: AgentInput):
    if not all([model, scaler, encoder]): raise HTTPException(503, "ML models not loaded.")
    try:
        base_details = get_soil_and_location_details(data.prompt)
        base_details = fill_missing_values_ai(base_details)
        city_for_weather = format_city_for_weather(base_details['city_or_state'])
        live_temp, live_humidity = get_live_weather(city_for_weather)
        final_data = {'N': base_details['N'], 'P': base_details['P'], 'K': base_details['K'], 'temperature': live_temp, 'humidity': live_humidity, 'ph': base_details['pH'], 'rainfall': base_details['rainfall']}
        return process_and_get_recommendations(final_data, base_details['city_or_state'])
    except Exception as e:
        raise HTTPException(500, f"An internal server error occurred: {e}")

@app.post("/predict/manual", tags=["Predictions"])
def predict_with_manual_input(data: ManualInput):
    if not all([model, scaler, encoder]): raise HTTPException(503, "ML models not loaded.")
    try:
        final_data = data.dict()
        location = final_data.pop('city_or_state', 'Unknown')
        return process_and_get_recommendations(final_data, location)
    except Exception as e:
        raise HTTPException(500, f"An internal server error occurred: {e}")

@app.post("/mini_chat", tags=["AI Assistant"])
def mini_chat_endpoint(data: MiniChatInput):
    try:
        # Split query by commas to detect multiple crops (if user enters multiple)
        crops = [c.strip() for c in data.query.split(",")]
        responses = {}
        
        for crop in crops:
            prompt = f"""
            As an expert Indian agronomist, answer concisely in 2-3 sentences.
            For a specific crop, give its ideal growing season and one important tip.
            Crop: "{crop}"
            """
            answer = groq_call_with_cache(f"mini_chat_{crop}", prompt, is_json=False)
            responses[crop] = {
                "human_readable": answer,
                "structured": {
                    "crop": crop,
                    "summary": answer
                }
            }
        
        return {"responses": responses}
    except Exception as e:
        raise HTTPException(500, f"Error processing chat message: {e}")


@app.get("/grow_guide/{crop_name}", tags=["Grow Guide"])
def get_grow_guide(crop_name: str):
    """Provides a detailed growing guide for a specific crop using an AI."""
    try:
        prompt = f"""
        You are an expert Indian agronomist providing a detailed guide for growing '{crop_name}' in India.
        Return ONLY a valid JSON object with the following keys:
        - "description": A short, engaging summary of the crop.
        - "season": The primary growing seasons (e.g., Kharif, Rabi) and ideal planting months.
        - "growth_duration": Typical time from sowing to harvest in days.
        - "irrigation_plan": A practical, brief irrigation schedule.
        - "pesticide_usage": Key pests/diseases and recommended management practices.
        """
        guide_data = groq_call_with_cache(f"grow_guide_{crop_name}", prompt, is_json=True)
        if not guide_data:
            raise HTTPException(status_code=404, detail="Could not generate a guide for this crop.")
        return guide_data
    except Exception as e:
        print(f"An error occurred in /grow_guide: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the grow guide.")



