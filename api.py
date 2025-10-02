# ------------------- FINAL SINGLE-FILE RESET CODE -------------------
import os
import json
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

# --- FastAPI App & Template Config ---
app = FastAPI(title="Crop Sathi API")
templates = Jinja2Templates(directory="templates")

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Load ML Models ---
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("✅ ML Models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load ML models. Error: {e}")
    model, scaler, encoder = None, None, None

# --- Pydantic Models for API ---
class MiniChatInput(BaseModel):
    query: str = Field(..., example="tell me about wheat")

# --- Main AI Function ---
def ask_groq_ai(system_prompt: str, user_prompt: str) -> str:
    if not GROQ_API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY environment variable is not set!")
        raise ValueError("API key for Groq is not configured.")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="gemma-7b-it",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"!!!!!!!!!! GROQ API CALL FAILED !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        raise e

# --- API Endpoints ---

# This endpoint serves your website's frontend
@app.get("/", response_class=HTMLResponse)
def serve_homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# This is your chatbot's API endpoint
@app.post("/mini_chat")
def mini_chat_endpoint(data: MiniChatInput):
    try:
        system_prompt = "You are a helpful Indian agronomist. Answer concisely in 2-3 sentences. For a specific crop, give its ideal growing season and one important tip."
        answer = ask_groq_ai(
            system_prompt=system_prompt,
            user_prompt=data.query
        )
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error communicating with the AI service. Check server logs.")

# Add other endpoints (like /predict/agent) here if you need them
