# ------------------- mini_kisan_copilot.py (Render-ready & Improved) -------------------

import os
import json
import requests
from groq import Groq

# --- Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_key_here")
CACHE_FILE = "groq_cache.json"

# --- Load cache ---
try:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            groq_cache = json.load(f)
    else:
        groq_cache = {}
except Exception as e:
    print(f"⚠️ Cache load failed: {e}")
    groq_cache = {}

# --- Groq API Wrapper with Caching ---
def groq_call_with_cache(key: str, prompt: str, is_json=True):
    key_lower = key.lower().replace(" ", "_")
    if key_lower in groq_cache:
        return groq_cache[key_lower]
    if not GROQ_API_KEY:
        return {} if is_json else "Groq key missing."
    try:
        client = Groq(api_key=GROQ_API_KEY)
        system_content = "You are a friendly Indian agronomist. Return JSON only." if is_json else "Plain text."
        response_format = {"type":"json_object"} if is_json else None
        resp = client.chat.completions.create(
            messages=[{"role":"system","content":system_content}, {"role":"user","content":prompt}],
            model="gemma2-9b-it",
            temperature=0.5,
            **({"response_format": response_format} if response_format else {})
        )
        content = resp.choices[0].message.content
        data = json.loads(content) if is_json else content
        groq_cache[key_lower] = data
        with open(CACHE_FILE, "w") as f: json.dump(groq_cache, f, indent=2)
        return data
    except Exception as e:
        print(f"⚠️ Groq call failed for {key}: {e}")
        return {} if is_json else f"Error: {e}"

# --- Live Crop Prices ---
def get_live_crop_prices():
    prompt = """
    Provide current average mandi prices (Rs/kg) of major Indian crops like rice, wheat, cotton, jute, coffee, mango.
    Return ONLY JSON in format: {"rice": 45, "wheat": 36, ...}
    """
    data = groq_call_with_cache("crop_prices", prompt)
    defaults = {"rice":45, "wheat":36, "cotton":80, "jute":60, "coffee":150, "mango":120}
    for k,v in defaults.items():
        if k not in data:
            data[k] = v
    return data

# --- Crop Dynamic Details ---
def get_crop_dynamic_details(crop_name: str):
    prompt = f"""
    You are an expert Indian agronomist.
    Provide structured JSON ONLY with keys:
    "definition", "soil", "irrigation", "fertilizer", "pesticides", "quick_tip".
    For "definition", explain clearly:
      - what the crop is
      - main uses
      - where it grows in India
      - why it is important
    Keep explanation short, friendly and clear.
    Crop: {crop_name}
    """
    data = groq_call_with_cache(f"crop_details_{crop_name}", prompt)
    defaults = {
        "definition": f"{crop_name.title()} is a widely grown crop in India, used for food and income. Major states include Punjab, Haryana, and Uttar Pradesh. Important for food security.",
        "soil":"Loamy or sandy loam soil with good drainage",
        "irrigation":"30-40 cm of water throughout the growing season",
        "fertilizer":"120-150 kg nitrogen, 60 kg phosphorus, 40 kg potassium per hectare",
        "pesticides":"Apply as needed based on pest/disease pressure",
        "quick_tip":"Maintain soil moisture and watch for pests."
    }
    for k in defaults:
        if k not in data:
            data[k] = defaults[k]
    return data

# --- Mini Co-pilot Response ---
def mini_copilot_response(user_query: str):
    crops = ["rice","wheat","cotton","jute","coffee","mango"]
    mentioned_crops = [crop for crop in crops if crop in user_query.lower()]

    if not mentioned_crops:
        return "🤖 Mini Kisan Co-pilot: Sorry bhai, main abhi sirf main crops ke info de sakta hu."

    prices = get_live_crop_prices()
    response = "🤖 Mini Kisan Co-pilot Report:\n\n"

    for crop in mentioned_crops:
        crop_info = get_crop_dynamic_details(crop)
        response += f"🌾 {crop.title()}:\n"
        response += f"1️⃣ Definition & Uses: {crop_info.get('definition','N/A')}\n"
        response += f"2️⃣ Price: Rs {prices.get(crop,'N/A')}/kg\n"
        response += f"3️⃣ Soil: {crop_info.get('soil','N/A')}\n"
        response += f"4️⃣ Irrigation: {crop_info.get('irrigation','N/A')}\n"
        response += f"5️⃣ Fertilizer: {crop_info.get('fertilizer','N/A')}\n"
        response += f"6️⃣ Pesticides: {crop_info.get('pesticides','N/A')}\n"
        response += f"7️⃣ Quick Tip: {crop_info.get('quick_tip','N/A')}\n\n"

    if len(mentioned_crops) > 1:
        response += "📊 Quick Price Comparison:\n"
        sorted_by_price = sorted(prices.items(), key=lambda x: x[1] if isinstance(x[1], (int,float)) else 0, reverse=True)
        for rank, (crop, price) in enumerate(sorted_by_price, start=1):
            response += f"{rank}. {crop.title()} → Rs {price}/kg\n"

    return response

# --- CLI ---
if __name__ == "__main__":
    print("=== Mini Kisan Co-pilot CLI (Render-ready, Gemma2-9B) ===")
    while True:
        query = input("\nAsk me about crops (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print(mini_copilot_response(query))
