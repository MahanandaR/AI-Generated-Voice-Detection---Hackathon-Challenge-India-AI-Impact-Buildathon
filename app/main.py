from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64

from app.config import API_KEY, SUPPORTED_LANGUAGES
from app.audio_utils import extract_features
from app.model_loader import predict

app = FastAPI(title="AI Generated Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    # API key validation
    if x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )

    # Language validation
    if request.language not in SUPPORTED_LANGUAGES:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )

    # Audio format validation
    if request.audioFormat.lower() != "mp3":
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        features = extract_features(audio_bytes)
        score = predict(features)

        classification = "AI_GENERATED" if score >= 0.5 else "HUMAN"

        explanation = (
            "Unnatural pitch consistency and robotic speech patterns detected"
            if classification == "AI_GENERATED"
            else "Natural pitch variation and human speech characteristics detected"
        )

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(score, 2),
            "explanation": explanation
        }

    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )
