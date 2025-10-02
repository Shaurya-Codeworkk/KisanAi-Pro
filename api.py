# ------------------- FINAL, MERGED api.py -------------------
import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Tuple

# --- Pydantic Models for API ---
class MiniChatInput(BaseModel):
    query: str = Field(..., example="tell me about wheat")

# --- FastAPI App & CORS ---
app = FastAPI(title="Crop Sathi API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Load ML Models ---
try:
    model = joblib.load('model_gbc.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    print("✅ ML Models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load ML models. Make sure .pkl files are uploaded. Error: {e}")
    model, scaler, encoder = None, None, None

# --- Main AI Function with Caching ---
def groq_call_with_cache(prompt_key: str, system_prompt: str, user_prompt: str) -> str:
    # This function now contains proper error checking
    if not GROQ_API_KEY:
        print("❌ CRITICAL: GROQ_API_KEY environment variable is not set!")
        raise ValueError("API key for Groq is not configured on the server.")

    # In a real app, you'd use a database like Redis for caching
    # For now, we'll just call the API directly every time on the server
    try:
        client = Groq(api_key=GROQ_API_KEY)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="gemma-7b-it",
            temperature=0.5,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        # This will print the *real* error to the logs (e.g., authentication, rate limit)
        print(f"!!!!!!!!!! GROQ API CALL FAILED !!!!!!!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Re-raise the exception to trigger our 500 error handler
        raise e

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Crop Sathi API is running."}

@app.post("/mini_chat", tags=["AI Assistant"])
def mini_chat_endpoint(data: MiniChatInput):
    if not all([model, scaler, encoder]):
        raise HTTPException(status_code=503, detail="ML models are not available.")
    try:
        system_prompt = "You are a helpful Indian agronomist. Answer concisely in 2-3 sentences. For a specific crop, give its ideal growing season and one important tip."
        answer = groq_call_with_cache(
            prompt_key=f"mini_chat_{data.query}",
            system_prompt=system_prompt,
            user_prompt=data.query
        )
        return {"response": answer}
    except Exception as e:
        # This is our main error handler that logs the issue.
        # The code in groq_call_with_cache will print the details.
        raise HTTPException(status_code=500, detail="An error occurred while communicating with the AI service. Check server logs.")
