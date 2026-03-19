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

model_path = Path(__file__).resolve().parent / "models"

def load_model(filename):
    model_file = model_path / filename
    if not model_file.exists():
        return None
    try:
        return YOLO(str(model_file))
    except Exception:
        return None


exit_model = load_model("exit_signs.pt")
crosswalk_on_model = load_model("crosswalk_on.pt")
crosswalk_off_model = load_model("crosswalk_off.pt")
crosswalk_model = load_model("crosswalk.pt")
school_sign_model = load_model("school_sign.pt")


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
        
@app.post("/signs")
async def signs(image: UploadFile = File(...)):
    if all(model_obj is None for model_obj in [exit_model, crosswalk_on_model, crosswalk_off_model, crosswalk_model, school_sign_model]):
        raise HTTPException(status_code = 500, detail = "No valid .pt model found in models folder")

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
            "distance": 0
        },
        "exit_right": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "exit_left": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "exit_both_ways": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "crosswalk": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "school_crosswalk": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "walk_on": {
            "detected": False,
            "direction": "center",
            "distance": 0
        },
        "walk_off": {
            "detected": False,
            "direction": "center",
            "distance": 0
        }
    }
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    best_area_by_sign = {}

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

    def normalize_sign_name(name):
        return name.lower().replace("-", "_").replace(" ", "_")

    def update_sign_with_box(sign_key, box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        if sign_key not in best_area_by_sign or area > best_area_by_sign[sign_key]:
            x_center = (x1 + x2) / 2
            frame_area = max(1, frame_width * frame_height)
            area_ratio = min(1.0, area / frame_area)
            detected_signs[sign_key]["detected"] = True
            detected_signs[sign_key]["direction"] = classify_direction(x_center, frame_width)
            detected_signs[sign_key]["distance"] = round((1.0 - area_ratio) * 100, 2)
            best_area_by_sign[sign_key] = area

    if exit_model is not None:
        exit_predictions = exit_model.predict(source = frame, verbose = False)
        exit_result = exit_predictions[0]
        for box in exit_result.boxes:
            class_id = int(box.cls[0])
            sign_key = id_to_sign.get(class_id)
            if sign_key in detected_signs:
                update_sign_with_box(sign_key, box)

    extra_model_targets = [
        (crosswalk_on_model, "walk_on"),
        (crosswalk_off_model, "walk_off"),
        (crosswalk_model, "crosswalk"),
        (school_sign_model, "school_crosswalk"),
    ]

    for model_obj, sign_key in extra_model_targets:
        if model_obj is None:
            continue
        prediction = model_obj.predict(source = frame, verbose = False)
        model_result = prediction[0]
        model_names = model_result.names
        for box in model_result.boxes:
            class_id = int(box.cls[0])
            class_name = model_names[class_id] if isinstance(model_names, dict) else model_names[class_id]
            if normalize_sign_name(class_name) != sign_key:
                continue
            update_sign_with_box(sign_key, box)
    

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