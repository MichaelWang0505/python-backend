import os
import tempfile
from pathlib import Path
import requests
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel
from groq import Groq
from ultralytics import YOLO

load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model_path = next((Path(__file__).resolve().parent / "models").glob("*.pt"), None)
model = YOLO(str(model_path)) if model_path else None

id_to_sign = {
    0: "exit_sign",
    1: "exit_right",
    2: "exit_left",
    3: "exit_both_ways",
}

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
async def signs(image: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code = 500, detail = "No .pt model found in models folder")

    raw_image = await image.read()
    if not raw_image:
        raise HTTPException(status_code = 400, detail = "No image")
    
    image_arr = np.frombuffer(raw_image, np.uint8)
    frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code = 400, detail = "Invalid image")
    
    detected_signs = {
        "exit_sign": {
            "detected": False,
            "direction": "center",
            "size": 0
        },
        "exit_right": {
            "detected": False,
            "direction": "center",
            "size": 0
        },
        "exit_left": {
            "detected": False,
            "direction": "center",
            "size": 0
        },
        "exit_both_ways": {
            "detected": False,
            "direction": "center",
            "size": 0
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
    
    predictions = model.predict(source = frame, verbose = False)
    result = predictions[0]
    names = result.names
    frame_width = frame.shape[1]
    best_conf_by_class = {}

    def classify_direction(x_center, width):
        ratio = (x_center / width) - 0.5
        distance = abs(ratio)
        if distance <= 0.10:
            return "center"
        if distance <= 0.20:
            return "slightly left" if ratio < 0 else "slightly right"
        if distance <= 0.35:
            return "left" if ratio < 0 else "right"
        return "far left" if ratio < 0 else "far right"

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_key = id_to_sign.get(class_id)
        if class_key is None:
            class_name = names[class_id]
            class_key = class_name.lower().replace("-", "_").replace(" ", "_")
        if class_key in detected_signs:
            detected_signs[class_key]["detected"] = True
            confidence = float(box.conf[0])
            if class_key not in best_conf_by_class or confidence > best_conf_by_class[class_key]:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = (x1 + x2) / 2
                if "direction" in detected_signs[class_key]:
                    detected_signs[class_key]["direction"] = classify_direction(x_center, frame_width)
                best_conf_by_class[class_key] = confidence

    return detected_signs
    
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