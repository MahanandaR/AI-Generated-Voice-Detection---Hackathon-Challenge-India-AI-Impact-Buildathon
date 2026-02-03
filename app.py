from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import tempfile
import os

from detector import analyze_voice

API_KEY = "guvi_voice_2026_key"
SUPPORTED_LANGUAGES = {"Tamil", "English", "Hindi", "Malayalam", "Telugu"}

app = FastAPI(title="AI Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def voice_detection(
    req: VoiceRequest,
    x_api_key: str = Header(None)
):
    # 🔐 API Key validation
    if x_api_key != API_KEY:
        return {"status": "error", "message": "Invalid API key"}

    # 📥 Input validation
    if req.language not in SUPPORTED_LANGUAGES:
        return {"status": "error", "message": "Unsupported language"}

    if req.audioFormat.lower() != "mp3":
        return {"status": "error", "message": "Only mp3 format supported"}

    if not req.audioBase64:
        return {"status": "error", "message": "Missing required fields"}

    # 🔁 Decode Base64 safely
    try:
        audio_bytes = base64.b64decode(req.audioBase64, validate=True)
    except Exception:
        return {"status": "error", "message": "Invalid Base64 data"}

    # 🗂️ Write MP3 temporarily (no modification)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    try:
        classification, confidence, explanation = analyze_voice(audio_path)
    finally:
        os.remove(audio_path)

    return {
        "status": "success",
        "language": req.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
