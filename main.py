import os
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq

load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RouteRequest(BaseModel):
    startLat: float
    startLon: float
    endLat: float
    endLon: float

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

groq_client = Groq(api_key = GROQ_API_KEY)

@app.get("/")
def root():
    return {"status": "good"}

@app.post("/voice_input")
async def voice_input(audio: UploadFile = File(...)):
    contents = await audio.read()
    
    with tempfile.NamedTemporaryFile(delete = False, suffix = ".m4a") as temp:
        temp.write(contents)
        temp_path = temp.name
        
    try:
        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file = (audio.filename or "audio.m4a", f),
                model = "whisper-large-v3",
            )
        return {"text": transcription.text}
    finally:
        os.unlink(temp_path)
        
@app.get("/signs")
def signs():
    return {
        "exit_sign": {
            "detected": False,
        },
        "exit_right": {
            "detected": False,
        },
        "exit_left": {
            "detected": False,
        },
        "exit_both_ways": {
            "detected": False,
        },
        "crosswalk": {
            "detected": False,
        },
        "school_crosswalk": {
            "detected": False,
        },
        "walk_on": {
            "detected": False,
        },
        "walk_off": {
            "detected": False,
        }
    }
    
@app.post("/api/route")
def get_route(route: RouteRequest):
    url = "https://api.openrouteservice.org/v2/directions/foot-walking/json"

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
    }

    body = {
        "coordinates": [
            [route.startLon, route.startLat],
            [route.endLon, route.endLat],
        ],
        "instructions": True,
    }

    try:
        response = requests.post(url, json=body, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        detail = response.text if "response" in locals() else "ORS HTTP error"
        raise HTTPException(status_code=response.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))