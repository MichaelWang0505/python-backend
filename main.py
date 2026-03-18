import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
import sounddevice as sd
import vosk
import json

load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")

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

@app.get("/")
def root():
    return {"status": "good"}

model = vosk.Model("vosk-model-small-en-us-0.15")

@app.get("/voice_input")
def voice_input():
    recognizer = vosk.KaldiRecognizer(model, 16000)
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1) as stream:
        while True:
            data, _ = stream.read(4000)
            if recognizer.AcceptWaveform(bytes(data)):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    return {"text": text}

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
    url = "https://api.openrouteservice.org/v2/directions/foot-walking"

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
        response = requests.post(url, json = body, headers = headers)
        return response.json()
    except Exception as e:
        return {"Error:", str(e)}