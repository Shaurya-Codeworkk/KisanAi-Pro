# ------------------- Yahan se Pura Final Merged Code Copy Karo -------------------

import joblib
import pandas as pd
import numpy as np
import requests
import json
from groq import Groq
from datetime import datetime
import os

# --- CONFIGURATION ---
import os

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") # Apni key yahan daalna
USE_GROQ = True
CACHE_FILE = "groq_cache.json"

# Load cache
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        groq_cache = json.load(f)
else:
    groq_cache = {}

print("Libraries import ho gayi hain...")

# --- Load ML Models ---
try:
    # --- YAHAN PAR TYPO FIX KIYA HAI ---
    model = joblib.load('model_gbc.pkl') 
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("ML Models successfully load ho gaye hain! ✅")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

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
    except:
        return raw_city_state.title()

def safe_json_parse(raw_content, fallback):
    try:
        return json.loads(raw_content)
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}")
        return fallback

# --- Live Weather ---
def get_live_weather(city_name):
    print(f"[Agent Research] {city_name} ka live weather dhoondh raha hai...")
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={WEATHER_API_KEY}&q={city_name}&units=metric"
    try:
        response = requests.get(complete_url, timeout=10)
        data = response.json()
        if data.get("cod") == 200:
            main = data["main"]
            temperature = main["temp"]
            humidity = main["humidity"]
            print(f"[Agent Research] Weather mil gaya: Temp={temperature}°C, Humidity={humidity}% ✅")
            return temperature, humidity
        else:
            print(f"⚠️ Weather API me '{city_name}' nahi mili, defaults use ho rahe hain.")
            return 28.0, 60.0
    except Exception as e:
        print(f"⚠️ Weather fetch error: {e}, defaults use ho rahe hain.")
        return 28.0, 60.0

# --- Groq API + Caching Wrapper ---
def groq_call_with_cache(key, prompt):
    key_lower = key.lower()
    if key_lower in groq_cache:
        return groq_cache[key_lower]
    if not USE_GROQ:
        print(f"[DEV MODE] Returning dummy data for {key}")
        return {}
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            messages=[{"role":"system","content":"Return valid JSON only."},
                      {"role":"user","content":prompt}],
            model="gemma2-9b-it",
            temperature=0.2,
            response_format={"type":"json_object"}
        )
        details = safe_json_parse(resp.choices[0].message.content, {})
        groq_cache[key_lower] = details
        with open(CACHE_FILE, "w") as f:
            json.dump(groq_cache, f)
        return details
    except Exception as e:
        print(f"⚠️ Groq fetch failed for {key}: {e}")
        return {}

# --- Soil & Location Interpreter ---
def get_soil_and_location_details(farmer_prompt):
    print("[Agent Reasoning] Farmer ki baat samajh raha hai...")
    prompt_template = f"""
        You are an expert Indian agronomist. Extract city/state and optional soil type from farmer prompt.
        Return ONLY JSON with keys: "city_or_state","soil_type".
        Farmer prompt: "{farmer_prompt}"
        """
    data = groq_call_with_cache(f"soil_location_{farmer_prompt}", prompt_template)
    if not data.get("city_or_state"): data["city_or_state"] = farmer_prompt
    if not data.get("soil_type"): data["soil_type"] = "unknown"
    print("[Agent Reasoning] Extract complete ✅", data)
    return data

# --- Fill Missing NPK/pH/rainfall ---
def fill_missing_values_ai(details):
    print("[AI Estimate] AI estimated values generate kar raha hai...")
    prompt = f"""
        You are an expert agronomist. Estimate N, P, K, pH, rainfall for given soil type and location.
        Return ONLY JSON with keys: "N","P","K","pH","rainfall".
        Location/soil: {details.get('city_or_state','')}, {details.get('soil_type','unknown')}
        """
    ai_values = groq_call_with_cache(f"npk_{details['city_or_state']}_{details['soil_type']}", prompt)
    defaults = {'N':50,'P':50,'K':50,'pH':6.5,'rainfall':400}
    for k in defaults:
        details[k] = float(ai_values.get(k, defaults[k]))
    print("[AI Estimate] Predicted NPK/pH/rainfall: ", {k: details[k] for k in ['N','P','K','pH','rainfall']})
    return details

# --- Prediction Function ---
def make_prediction(input_data, top_n=5):
    df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    scaled_features = scaler.transform(df)
    probs = model.predict_proba(scaled_features)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    results = []
    for idx in top_indices:
        crop = encoder.inverse_transform([idx])[0].lower()
        results.append((crop, round(probs[idx]*100,2)))
    return results

# --- Fetch Live Crop Prices ---
def get_live_crop_prices():
    prompt = """
        Provide current average mandi prices (Rs/kg) of major Indian crops like rice, wheat, cotton, jute, coffee, mango, pigeonpeas.
        Return ONLY JSON in format: {"rice": 45, "wheat": 36, ...}
        """
    prices = groq_call_with_cache("crop_prices", prompt)
    defaults = {"rice":40, "wheat":35, "cotton":80, "jute":60, "coffee":150, "mango":120, "pigeonpeas": 90}
    for k,v in defaults.items():
        if k not in prices: prices[k] = v
    return {k.lower(): float(v) for k,v in prices.items()}

# --- NAYA FEATURE: AI FUTURE PRICE PREDICTION ---
def get_future_price_ai(crop_name, location):
    print(f"\n[AI Price Predictor] {location} mein '{crop_name}' ka future price predict kar raha hai...")
    prompt = f"""
        You are an expert Indian market analyst.
        Based on current trends, predict the approximate price (Rs/kg) of '{crop_name}' in the '{location}' region after a typical 6-month harvest period.
        Consider seasonal demand and supply factors.
        Return ONLY JSON with the key: "future_price". The value should be a number.
    """
    prediction = groq_call_with_cache(f"future_price_{crop_name}_{location}", prompt)
    return float(prediction.get("future_price", -1.0)) # -1 agar fail ho

# --- AI Dynamic Crop Details Fetcher ---
def get_crop_dynamic_details(crop_name):
    prompt = f"""
        You are an agronomist. Provide soil type, irrigation, fertilizer, pesticides for {crop_name}.
        Return JSON ONLY with keys: "soil", "irrigation", "fertilizer", "pesticides"
        """
    details = groq_call_with_cache(f"crop_details_{crop_name}", prompt)
    defaults = {"soil":"loamy", "irrigation":"flooding", "fertilizer":"NPK 20-20-20", "pesticides":"2kg/ha"}
    for k in defaults:
        if k not in details: details[k] = defaults[k]
    return details
    
# --- Smart Crop Rotation ---
def get_crop_rotation_plan(current_crop, location):
    print(f"\n[AI Rotation Planner] {location} mein '{current_crop}' ke baad ka plan generate kar raha hai...")
    prompt = f"""
        You are an expert Indian agronomist specializing in sustainable farming.
        A farmer in {location}, India just harvested '{current_crop}'.
        Suggest a smart 2-season crop rotation plan to improve soil health and income.
        Provide a brief reason for each choice.
        Return ONLY JSON with keys: "season_1_crop", "season_1_reason", "season_2_crop", "season_2_reason".
        """
    plan = groq_call_with_cache(f"rotation_{current_crop}_{location}", prompt)
    if "season_1_crop" not in plan:
        return {"error": "Could not generate a plan."}
    print("[AI Rotation Planner] Plan ready! ✅")
    return plan

# --- Top 3 Recommendation Logic (Updated) ---
def rank_top_3(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    if not crop_probs:
        return ["N/A", "N/A", "N/A"]
        
    # Revenue ke liye future price use karo, agar nahi hai toh live price use karo
    sorted_by_revenue = sorted(crop_probs, key=lambda x: future_prices.get(x[0], live_prices.get(x[0], 0)), reverse=True)
    sorted_by_transport = sorted(crop_probs, key=lambda x: transport_score.get(x[0],0), reverse=True)
    
    balanced_score = []
    for crop, prob in crop_probs:
        rev_score = future_prices.get(crop, live_prices.get(crop, 0))
        trans_score = transport_score.get(crop, 0)
        score = (prob * 0.5) + (rev_score * 0.3) + (trans_score * 0.2)
        balanced_score.append((crop, score))

    sorted_balanced = sorted(balanced_score, key=lambda x:x[1], reverse=True)
    
    return [
        sorted_by_revenue[0][0] if sorted_by_revenue else "N/A",
        sorted_by_transport[0][0] if sorted_by_transport else "N/A",
        sorted_balanced[0][0] if sorted_balanced else "N/A"
    ]

# --- Traffic Light Indicator ---
def traffic_light_color(suitability):
    if suitability >= 70: return "Green"
    elif suitability >= 40: return "Yellow"
    else: return "Red"

# --- Display Multi-Crop Comparison (Updated) ---
def display_crop_table(crop_probs, live_prices, future_prices):
    transport_score = {"rice": 80, "wheat": 85, "cotton": 50, "jute": 60, "coffee": 40, "mango": 30, "pigeonpeas": 70}
    print("\n📊 Multi-Crop Comparison Table:")
    print("Crop       | Suitability | Live Price | Future Price | Transport | Color")
    print("-----------------------------------------------------------------------------")
    for crop, prob in crop_probs:
        live = live_prices.get(crop, 'N/A')
        future_val = future_prices.get(crop, -1.0)
        future = future_val if future_val > 0 else 'N/A'
        transport = transport_score.get(crop,0)
        color = traffic_light_color(prob)
        print(f"{crop.title():<10} | {prob:>10}% | Rs {live:<7} | Rs {future:<10} | {transport:>9} | {color}")

# --- Save Results ---
def save_results(input_data, predictions):
    try:
        row = {**input_data, "predictions": str(predictions), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        df = pd.DataFrame([row])
        file_exists = os.path.exists("prediction_logs.csv")
        df.to_csv("prediction_logs.csv", mode="a", header=not file_exists, index=False)
        print("📝 Results logged into prediction_logs.csv ✅")
    except Exception as e:
        print(f"⚠️ Logging failed: {e}")

# --- Main Workflow ---
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
                print(f"  → Agla Season: {rotation_plan.get('season_1_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_1_reason', 'N/A')})")
                print(f"  → Uske Baad: {rotation_plan.get('season_2_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_2_reason', 'N/A')})")
            
            save_results(final_data, top_crops)
    else:
        # --- Manual Data Entry ---
        print("\n📥 Manual Data Entry Mode")
        try:
            city_or_state = input("City/State: ").strip()
            soil_type = input("Soil Type: ").strip() or "unknown"
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
                
                future_prices = {}
                for crop, _ in top_crops:
                    future_prices[crop] = get_future_price_ai(crop, city_or_state)

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
                    print(f"  → Agla Season: {rotation_plan.get('season_1_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_1_reason', 'N/A')})")
                    print(f"  → Uske Baad: {rotation_plan.get('season_2_crop', 'N/A').title()} (Reason: {rotation_plan.get('season_2_reason', 'N/A')})")

                save_results(final_data, top_crops)
        except Exception as e:
            print(f"❌ Manual input failed: {e}")

# ------------------- Yahan tak Copy Karo -------------------