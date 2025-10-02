import joblib
import pandas as pd
import numpy as np
import requests
import json
from groq import Groq
from datetime import datetime
import os

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
CACHE_FILE = "groq_cache.json"

# Load cache
# Note: On Render's free tier, this cache file will be deleted when the server sleeps.
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            groq_cache = json.load(f)
    else:
        groq_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    groq_cache = {}

print("Libraries import ho gayi hain...")

# --- Load ML Models ---
try:
    model = joblib.load('model_gbc.pkl') 
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ CRITICAL Error loading models: {e}. The app might not work correctly.")
    # Removed exit() to prevent server crash

# --- State Abbreviations Mapping ---
state_map = {
    "up": "Uttar Pradesh", "mh": "Maharashtra", "hr": "Haryana",
    "ka": "Karnataka", "tn": "Tamil Nadu",
}

# --- Helper Functions ---
def format_city_for_weather(raw_city_state):
    try:
        parts = [p.strip() for p in raw_city_state.split(",")]
        city = parts[0].title()
        state = parts[1].lower() if len(parts) > 1 else ""
        state_full = state_map.get(state, state.title())
        return f"{city},{state_full},IN"
    except Exception:
        return raw_city_state.title()

def safe_json_parse(raw_content, fallback):
    try:
        # Added logic to handle markdown code blocks in AI response
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0]
        return json.loads(raw_content)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}. Content was: {raw_content}")
        return fallback

# --- Live Weather ---
def get_live_weather(city_name):
    print(f"[Agent Research] Searching for live weather in {city_name}...")
    if not WEATHER_API_KEY:
        print("⚠️ WEATHER_API_KEY is not set. Using default weather values.")
        return 28.0, 60.0
        
    base_url = "[https://api.openweathermap.org/data/2.5/weather](https://api.openweathermap.org/data/2.5/weather)?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
    try:
        response = requests.get(complete_url, timeout=10)
        response.raise_for_status() # Will raise an error for bad responses (4xx or 5xx)
        data = response.json()
        if data.get("cod") == 200:
            main = data["main"]
            temperature = main["temp"]
            humidity = main["humidity"]
            print(f"[Agent Research] Weather found: Temp={temperature}°C, Humidity={humidity}% ✅")
            return temperature, humidity
        else:
            print(f"⚠️ Weather API could not find '{city_name}', using defaults.")
            return 28.0, 60.0
    except Exception as e:
        print(f"⚠️ Weather fetch error: {e}, using defaults.")
        return 28.0, 60.0

# --- NEW Upgraded Groq API Wrapper ---
def ask_groq_ai(prompt_key: str, system_prompt: str, user_prompt: str):
    key_lower = prompt_key.lower().replace(" ", "_")
    if key_lower in groq_cache:
        return groq_cache[key_lower]
        
    if not GROQ_API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY environment variable is not set!")
        return {} # Return empty dict on failure
        
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",  # <-- UPGRADED MODEL NAME
            temperature=0.3, # Slightly lower temp for more factual results
            response_format={"type": "json_object"}
        )
        content = chat_completion.choices[0].message.content
        data = safe_json_parse(content, {})
        
        if data: # Only cache if parsing was successful
            groq_cache[key_lower] = data
            with open(CACHE_FILE, "w") as f:
                json.dump(groq_cache, f, indent=2)
                
        return data
    except Exception as e:
        print(f"!!!!!!!!!! GROQ API CALL FAILED for key: {prompt_key} !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return {}

# --- Soil & Location Interpreter ---
def get_soil_and_location_details(farmer_prompt):
    print("[Agent Reasoning] Understanding farmer's query...")
    system_prompt = "You are an expert Indian agronomist. Your task is to extract the city/state and an optional soil type from a farmer's query. Return ONLY a valid JSON object with keys: 'city_or_state' and 'soil_type'."
    user_prompt = f"Farmer query: '{farmer_prompt}'"
    
    data = ask_groq_ai(f"soil_location_{farmer_prompt}", system_prompt, user_prompt)
    
    if not data.get("city_or_state"): data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"): data["soil_type"] = "unknown"
    
    print("[Agent Reasoning] Extraction complete ✅", data)
    return data

# --- Fill Missing NPK/pH/rainfall ---
def fill_missing_values_ai(details):
    print("[AI Estimate] Generating estimated values...")
    location_soil = f"{details.get('city_or_state', '')}, {details.get('soil_type', 'unknown')}"
    
    system_prompt = "You are an expert agronomist. Your task is to estimate the typical N, P, K, pH, and annual rainfall (in mm) for a given soil type and location in India. Return ONLY a valid JSON object with numeric values for keys: 'N', 'P', 'K', 'pH', 'rainfall'."
    user_prompt = f"Estimate for: {location_soil}"

    ai_values = ask_groq_ai(f"npk_{location_soil}", system_prompt, user_prompt)
    
    defaults = {'N': 50.0, 'P': 50.0, 'K': 50.0, 'pH': 6.5, 'rainfall': 400.0}
    for k in defaults:
        details[k] = float(ai_values.get(k, defaults[k]))
        
    print("[AI Estimate] Predicted NPK/pH/rainfall: ", {k: details[k] for k in ['N','P','K','pH','rainfall']})
    return details

# --- Prediction Function (No changes needed here) ---
def make_prediction(input_data, top_n=5):
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled_features = scaler.transform(df)
    probs = model.predict_proba(scaled_features)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    results = []
    for idx in top_indices:
        crop = encoder.inverse_transform([idx])[0].lower()
        results.append((crop, round(probs[idx]*100, 2)))
    return results

# --- Fetch Live Crop Prices ---
def get_live_crop_prices():
    print("[Market Research] Fetching live crop prices...")
    system_prompt = "You are a market data provider. Provide current average mandi prices in India (in Rs/kg) for major crops. Return ONLY a valid JSON object in the format: {'rice': 45, 'wheat': 36, ...}"
    user_prompt = "Provide prices for: rice, wheat, cotton, jute, coffee, mango, pigeonpeas."

    prices = ask_groq_ai("crop_prices", system_prompt, user_prompt)
    
    defaults = {"rice": 40.0, "wheat": 35.0, "cotton": 80.0, "jute": 60.0, "coffee": 150.0, "mango": 120.0, "pigeonpeas": 90.0}
    for k, v in defaults.items():
        if k not in prices: prices[k] = v
        
    return {k.lower(): float(v) for k, v in prices.items()}

# --- AI FUTURE PRICE PREDICTION ---
def get_future_price_ai(crop_name, location):
    print(f"[AI Price Predictor] Predicting future price for '{crop_name}' in {location}...")
    system_prompt = "You are an expert Indian market analyst. Your task is to predict the approximate price (in Rs/kg) of a crop in a specific region after a typical 6-month harvest period, considering seasonal trends. Return ONLY a valid JSON object with the key 'future_price' and a numeric value."
    user_prompt = f"Predict the 6-month future price of '{crop_name}' in the '{location}' region."

    prediction = ask_groq_ai(f"future_price_{crop_name}_{location}", system_prompt, user_prompt)
    
    return float(prediction.get("future_price", -1.0)) # -1 if it fails

# --- AI Dynamic Crop Details Fetcher (Not used in main workflow but kept as a feature) ---
def get_crop_dynamic_details(crop_name):
    print(f"[AI Details] Fetching details for {crop_name}...")
    system_prompt = "You are an agronomist. Provide structured data for a crop. Return ONLY a valid JSON object with keys: 'soil', 'irrigation', 'fertilizer', 'pesticides'."
    user_prompt = f"Provide details for {crop_name}."

    details = ask_groq_ai(f"crop_details_{crop_name}", system_prompt, user_prompt)
    
    defaults = {"soil": "loamy", "irrigation": "flooding", "fertilizer": "NPK 20-20-20", "pesticides": "2kg/ha"}
    for k in defaults:
        if k not in details: details[k] = defaults[k]
    return details
    
# --- Smart Crop Rotation ---
def get_crop_rotation_plan(current_crop, location):
    print(f"[AI Rotation Planner] Generating plan for '{current_crop}' in {location}...")
    system_prompt = "You are an expert Indian agronomist specializing in sustainable farming. Suggest a smart 2-season crop rotation plan after a harvest. Provide a brief reason for each choice. Return ONLY a valid JSON object with keys: 'season_1_crop', 'season_1_reason', 'season_2_crop', 'season_2_reason'."
    user_prompt = f"A farmer in {location}, India just harvested '{current_crop}'. Suggest the next two seasons of crops."

    plan = ask_groq_ai(f"rotation_{current_crop}_{location}", system_prompt, user_prompt)
    
    if "season_1_crop" not in plan:
        return {"error": "Could not generate a plan."}
        
    print("[AI Rotation Planner] Plan ready! ✅")
    return plan

# --- Top 3 Recommendation Logic (No changes needed) ---
def rank_top_3(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    if not crop_probs:
        return ["N/A", "N/A", "N/A"]
        
    sorted_by_revenue = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    sorted_by_transport = sorted(crop_probs, key=lambda x: transport_score.get(x[0], 0), reverse=True)
    
    balanced_score = []
    for crop, prob in crop_probs:
        rev_score = future_prices.get(crop, live_prices.get(crop, 0))
        trans_score = transport_score.get(crop, 0)
        score = (prob * 0.5) + (rev_score * 0.3) + (trans_score * 0.2)
        balanced_score.append((crop, score))

    sorted_balanced = sorted(balanced_score, key=lambda x: x[1], reverse=True)
    
    return [
        sorted_by_revenue[0][0] if sorted_by_revenue else "N/A",
        sorted_by_transport[0][0] if sorted_by_transport else "N/A",
        sorted_balanced[0][0] if sorted_balanced else "N/A"
    ]

# --- Traffic Light Indicator (No changes needed) ---
def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    elif suitability >= 40: return "Yellow"
    else: return "Red"

# --- Display Multi-Crop Comparison (For CLI testing, no changes needed) ---
def display_crop_table(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    print("\n📊 Multi-Crop Comparison Table:")
    print("Crop       | Suitability | Live Price | Future Price | Transport | Color")
    print("-----------------------------------------------------------------------------")
    for crop, prob in crop_probs:
        live = live_prices.get(crop, 'N/A')
        future_val = future_prices.get(crop, -1.0)
        future = future_val if future_val > 0 else 'N/A'
        transport = transport_score.get(crop, 0)
        color = traffic_light_color(prob)
        print(f"{crop.title():<10} | {prob:>10}% | Rs {live:<7} | Rs {future:<10} | {transport:>9} | {color}")

# --- Save Results (No changes needed) ---
def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        file_exists = os.path.exists("prediction_logs.csv")
        df.to_csv("prediction_logs.csv", mode="a", header=not file_exists, index=False)
        print("📝 Results logged into prediction_logs.csv ✅")
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")

# --- Main Workflow (For CLI testing, no changes needed) ---
if __name__ == "__main__":
    choice = input("\nChoose input method:\n1 → AI Agent\n2 → Manual Data Entry\nEnter 1 or 2: ")
    if choice.strip() == "1":
        farmer_input = input("Farmer prompt daalein (village/state + soil type optional): ")
        base_details = get_soil_and_location_details(farmer_input)
        base_details = fill_missing_values_ai(base_details)

        city_for_weather = format_city_for_weather(base_details['city_or_state'])
        live_temp, live_humidity = get_live_weather(city_for_weather)

        final_data = {
            'N': base_details['N'], 'P': base_details['P'], 'K': base_details['K'],
            'temperature': live_temp, 'humidity': live_humidity,
            'ph': base_details['pH'], 'rainfall': base_details['rainfall']
        }

        top_crops = make_prediction(final_data, top_n=5)
        if top_crops:
            live_prices = get_live_crop_prices()
            
            future_prices = {}
            for crop, _ in top_crops:
                future_prices[crop] = get_future_price_ai(crop, base_details['city_or_state'])

            display_crop_table(top_crops, live_prices, future_prices)
            
            ranked_top3 = rank_top_3(top_crops, live_prices, future_prices)
            print("\n🚀 Top 3 Focused Recommendations:")
            print(f"1️⃣ Best Revenue (Future Price) → {ranked_top3[0].title()}")
            print(f"2️⃣ Transport-friendly → {ranked_top3[1].title()}")
            print(f"3️⃣ Balanced → {ranked_top3[2].title()}")
            
            best_crop_for_rotation = ranked_top3[2] 
            rotation_plan = get_crop_rotation_plan(best_crop_for_rotation, base_details['city_or_state'])
            if "error" not in rotation_plan:
                print("\n🌿 Smart Crop Rotation Plan:")
                print(f"  → Agla Season: {rotation_plan.get('season_1_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_1_reason', 'N/A')})")
                print(f"  → Uske Baad: {rotation_plan.get('season_2_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_2_reason', 'N/A')})")
            
            save_results(final_data, top_crops)
    else:
        # --- Manual Data Entry ---
        print("\n📥 Manual Data Entry Mode")
        try:
            city_or_state = input("City/State: ").strip()
            N = float(input("Nitrogen (N): "))
            P = float(input("Phosphorus (P): "))
            K = float(input("Potassium (K): "))
            pH = float(input("pH value: "))
            rainfall = float(input("Rainfall (mm/year): "))
            temperature = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))

            final_data = {
                'N': N, 'P': P, 'K': K,
                'temperature': temperature, 'humidity': humidity,
                'ph': pH, 'rainfall': rainfall
            }

            top_crops = make_prediction(final_data, top_n=5)
            if top_crops:
                live_prices = get_live_crop_prices()
                future_prices = {crop: get_future_price_ai(crop, city_or_state) for crop, _ in top_crops}
                display_crop_table(top_crops, live_prices, future_prices)
                ranked_top3 = rank_top_3(top_crops, live_prices, future_prices)
                print("\n🚀 Top 3 Focused Recommendations:")
                print(f"1️⃣ Best Revenue (Future Price) → {ranked_top3[0].title()}")
                print(f"2️⃣ Transport-friendly → {ranked_top3[1].title()}")
                print(f"3️⃣ Balanced → {ranked_top3[2].title()}")
                best_crop_for_rotation = ranked_top3[2]
                rotation_plan = get_crop_rotation_plan(best_crop_for_rotation, city_or_state)
                if "error" not in rotation_plan:
                    print("\n🌿 Smart Crop Rotation Plan:")
                    print(f"  → Agla Season: {rotation_plan.get('season_1_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_1_reason', 'N/A')})")
                    print(f"  → Uske Baad: {rotation_plan.get('season_2_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_2_reason', 'N/A')})")
                save_results(final_data, top_crops)
        except Exception as e:
            print(f"❌ Manual input failed: {e}")
