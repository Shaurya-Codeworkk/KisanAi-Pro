# ------------------- API Merged Code -------------------
# To run this API:
# 1. Install necessary packages: pip install fastapi uvicorn python-multipart joblib pandas numpy requests groq
# 2. Run the server from your terminal: uvicorn api:app --reload

import joblib
import pandas as pd
import numpy as np
import requests
import json
from groq import Groq
from datetime import datetime
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Tuple

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

# --- Helper Functions ---
def format_city_for_weather(raw_city_state: str) -> str:
    try:
        parts = [p.strip() for p in raw_city_state.split(",")]
        city = parts[0].title()
        state_abbr = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state_abbr, state_abbr.title())
        return f"{city},{state_full},IN"
    except Exception:
        return raw_city_state.title()

def safe_json_parse(raw_content: str, fallback: Dict) -> Dict:
    try:
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0]
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            return json.loads(raw_content[json_start:json_end])
        return fallback
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content}")
        return fallback

# --- Core Logic Functions ---
def get_live_weather(city_name: str) -> Tuple[float, float]:
    base_url = "[https://api.openweathermap.org/data/2.5/weather](https://api.openweathermap.org/data/2.5/weather)?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status() 
        data = response.json()
        if data.get("cod") == 200:
            main = data["main"]
            return main.get("temp", 28.0), main.get("humidity", 60.0)
        return 28.0, 60.0
    except requests.exceptions.RequestException:
        return 28.0, 60.0

def groq_call_with_cache(key: str, prompt: str, is_json: bool = True) -> Any:
    key_lower = key.lower().replace(" ", "_").replace('"', '')
    if key_lower in groq_cache: return groq_cache[key_lower]
    if not USE_GROQ: return {} if is_json else "Groq is disabled."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = "You are a helpful assistant that MUST return responses in valid JSON format only." if is_json else "You are a friendly Indian agronomist. Provide clear, concise answers in plain text about farming topics."
        response_format = {"type": "json_object"} if is_json else None
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
        params = {"messages": messages, "model": "gemma2-9b-it", "temperature": 0.3}
        if response_format: params["response_format"] = response_format
        resp = client.chat.completions.create(**params)
        content = resp.choices[0].message.content
        result = safe_json_parse(content, {}) if is_json else content
        if result:
            groq_cache[key_lower] = result
            with open(CACHE_FILE, "w") as f: json.dump(groq_cache, f, indent=4)
        return result
    except Exception as e:
        print(f"⚠️ Groq API call failed for {key}: {e}")
        return {} if is_json else "Sorry, an error occurred."

def get_soil_and_location_details(prompt: str) -> Dict[str, str]:
    p_template = f'Extract city/state and soil type from: "{prompt}". Return JSON with "city_or_state", "soil_type".'
    data = groq_call_with_cache(f"soil_loc_{prompt}", p_template)
    if not data.get("city_or_state"): data["city_or_state"] = prompt
    if not data.get("soil_type"): data["soil_type"] = "unknown"
    return data

def fill_missing_values_ai(details: Dict[str, Any]) -> Dict[str, Any]:
    loc_soil = f"{details.get('city_or_state','')}, {details.get('soil_type','unknown')}"
    prompt = f'Estimate typical N, P, K, pH, and rainfall (mm) for {loc_soil} in India. Return JSON with numeric values for "N", "P", "K", "pH", "rainfall".'
    ai_values = groq_call_with_cache(f"npk_{loc_soil}", prompt)
    defaults = {'N': 90.0, 'P': 42.0, 'K': 43.0, 'pH': 6.5, 'rainfall': 100.0}
    for k in defaults: details[k] = float(ai_values.get(k, defaults[k]))
    return details

def make_prediction(data: Dict[str, float], top_n: int = 5) -> List[Tuple[str, float]]:
    if not all([model, scaler, encoder]): raise RuntimeError("ML models not loaded.")
    df = pd.DataFrame([data], columns=model.feature_names_in_)
    probs = model.predict_proba(scaler.transform(df))[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    return [(encoder.inverse_transform([idx])[0].lower(), round(probs[idx] * 100, 2)) for idx in top_indices]

def get_live_crop_prices() -> Dict[str, float]:
    prompt = 'Current avg mandi prices (Rs/quintal) in India for: rice, wheat, cotton, jute, coffee, mango, pigeonpeas, orange, coconut, potato, maize, mothbeans. Return JSON: {"rice": 4500, ...}'
    prices_per_quintal = groq_call_with_cache("crop_prices_quintal", prompt)
    defaults_quintal = {"rice": 4000, "wheat": 3500, "cotton": 8000, "jute": 6000, "coffee": 15000, "mango": 12000, "pigeonpeas": 9000, "orange": 5000, "coconut": 3000, "potato": 2000, "maize": 2200, "mothbeans": 7500}
    
    prices_per_kg = {}
    for k, v in defaults_quintal.items():
        price = prices_per_quintal.get(k, v)
        prices_per_kg[k.lower()] = float(price) / 100
        
    return prices_per_kg

def get_future_price_ai(crop: str, loc: str) -> float:
    prompt = f'Predict avg price (Rs/quintal) of "{crop}" in "{loc}" after 6 months. Consider trends, demand, inflation. Return JSON: {{"future_price":_number_}}.'
    prediction = groq_call_with_cache(f"future_price_quintal_{crop}_{loc}", prompt)
    price_quintal = float(prediction.get("future_price", -1.0))
    return price_quintal / 100 if price_quintal > 0 else -1.0

def get_crop_rotation_plan(crop: str, loc: str) -> Dict[str, str]:
    prompt = f'Farmer in {loc}, India harvested "{crop}". Suggest a smart 2-season crop rotation plan for soil health & income. Give brief reasons. Return JSON: {{"season_1_crop":"", "season_1_reason":"", "season_2_crop":"", "season_2_reason":""}}.'
    plan = groq_call_with_cache(f"rotation_{crop}_{loc}", prompt)
    if "season_1_crop" not in plan: return {"error": "Could not generate plan."}
    return plan

def rank_top_3(crop_probs: List[Tuple[str, float]], live_prices: Dict, future_prices: Dict) -> Dict[str, str]:
    transport_score = { "rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70, "orange": 45, "maize": 75, "potato": 65, "coconut": 25, "mothbeans": 70 }
    if not crop_probs: return {"best_revenue": "N/A", "transport_friendly": "N/A", "balanced_choice": "N/A"}
    
    by_revenue = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    by_transport = sorted(crop_probs, key=lambda x: transport_score.get(x[0], 30), reverse=True)
    
    balanced_score = [
        (c, (p * 0.6) + (future_prices.get(c, live_prices.get(c, 0)) * 0.2) + (transport_score.get(c, 30) * 0.2)) for c, p in crop_probs
    ]
    sorted_balanced = sorted(balanced_score, key=lambda x: x[1], reverse=True)
    
    return {
        "best_revenue": by_revenue[0][0] if by_revenue else "N/A",
        "transport_friendly": by_transport[0][0] if by_transport else "N/A",
        "balanced_choice": sorted_balanced[0][0] if sorted_balanced else "N/A"
    }

def traffic_light_color(s: float) -> str:
    if s >= 70: return "Green"
    if s >= 40: return "Yellow"
    return "Red"

# --- Main Processing Function ---
def process_and_get_recommendations(final_data: Dict[str, float], location: str) -> Dict[str, Any]:
    top_crops = make_prediction(final_data, top_n=5)
    if not top_crops: raise HTTPException(status_code=404, detail="Could not generate crop predictions.")
    
    live_prices = get_live_crop_prices()
    future_prices = {crop: get_future_price_ai(crop, location) for crop, _ in top_crops}
    
    comparison_table = [
        {"crop": c.title(), "suitability_percent": p, "live_price_rs_kg": f'₹{live_prices.get(c, 0):.2f}',
         "predicted_future_price_rs_kg": f'₹{future_prices.get(c, -1.0):.2f}' if future_prices.get(c, -1.0) > 0 else 'N/A',
         "recommendation_color": traffic_light_color(p)} for c, p in top_crops]
    
    ranked_top3 = rank_top_3(top_crops, live_prices, future_prices)
    rotation_plan = {}
    if ranked_top3["balanced_choice"] != "N/A":
        rotation_plan = get_crop_rotation_plan(ranked_top3["balanced_choice"], location)
    
    return {"input_parameters": final_data, "location_analyzed": location, "comparison_table": comparison_table, "top_3_recommendations": ranked_top3, "smart_rotation_plan": rotation_plan}

# --- API Endpoints ---
@app.get("/")
def read_root(): return {"status": "ok"}

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

