import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from ultralytics import YOLO

load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --------------------------
# Config
# --------------------------
MAX_IMAGE_SIDE = 960      # Resize input before inference
YOLO_IMGSZ = 640          # Inference size
YOLO_CONF = 0.25          # Confidence threshold
YOLO_IOU = 0.45           # NMS IoU threshold

MODEL_DIR = Path(__file__).resolve().parent / "models"

# --------------------------
# FastAPI app
# --------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep a session for outbound ORS calls
http_session = requests.Session()


class RouteRequest(BaseModel):
    startLat: float
    startLon: float
    endLat: float
    endLon: float


def load_model(filename: str) -> Optional[YOLO]:
    model_file = MODEL_DIR / filename
    if not model_file.exists():
        print(f"[model] Missing: {model_file}")
        return None
    try:
        model = YOLO(str(model_file))
        print(f"[model] Loaded: {filename}")
        return model
    except Exception as exc:
        print(f"[model] Failed loading {filename}: {exc}")
        return None


exit_model = load_model("exit_signs.pt")
crosswalk_on_model = load_model("crosswalk_on.pt")
crosswalk_off_model = load_model("crosswalk_off.pt")
crosswalk_model = load_model("crosswalk.pt")
school_sign_model = load_model("school_sign.pt")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

id_to_sign = {
    0: "exit_sign",
    1: "exit_right",
    2: "exit_left",
    3: "exit_both_ways",
}


def empty_detected_signs() -> Dict[str, Dict[str, object]]:
    return {
        "exit_sign": {"detected": False, "direction": "center", "distance": 0},
        "exit_right": {"detected": False, "direction": "center", "distance": 0},
        "exit_left": {"detected": False, "direction": "center", "distance": 0},
        "exit_both_ways": {"detected": False, "direction": "center", "distance": 0},
        "crosswalk": {"detected": False, "direction": "center", "distance": 0},
        "school_crosswalk": {"detected": False, "direction": "center", "distance": 0},
        "walk_on": {"detected": False, "direction": "center", "distance": 0},
        "walk_off": {"detected": False, "direction": "center", "distance": 0},
    }


def preprocess_frame(raw_image: bytes) -> np.ndarray:
    if not raw_image:
        raise HTTPException(status_code=400, detail="No image")

    image_arr = np.frombuffer(raw_image, np.uint8)
    frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    h, w = frame.shape[:2]
    max_side = max(h, w)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return frame


def classify_direction(x_center: float, width: int) -> str:
    ratio = (x_center / width) - 0.5
    distance = abs(ratio)
    if distance <= 0.10:
        return "center"
    if distance <= 0.20:
        return "slightly left" if ratio < 0 else "slightly right"
    if distance <= 0.35:
        return "left" if ratio < 0 else "right"
    return "far left" if ratio < 0 else "far right"


def normalize_sign_name(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def run_predict(model: YOLO, frame: np.ndarray):
    # Using explicit settings for more stable latency
    return model.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        verbose=False,
        max_det=10,
    )


def update_sign_with_box(
    detected_signs: Dict[str, Dict[str, object]],
    best_area_by_sign: Dict[str, float],
    sign_key: str,
    box,
    frame_width: int,
    frame_height: int,
) -> None:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    if sign_key not in best_area_by_sign or area > best_area_by_sign[sign_key]:
        x_center = (x1 + x2) / 2.0
        frame_area = max(1, frame_width * frame_height)
        area_ratio = min(1.0, area / frame_area)

        detected_signs[sign_key]["detected"] = True
        detected_signs[sign_key]["direction"] = classify_direction(x_center, frame_width)
        detected_signs[sign_key]["distance"] = round((1.0 - area_ratio) * 100, 2)

        best_area_by_sign[sign_key] = area


@app.on_event("startup")
def warmup_models():
    # Warmup reduces first-request latency spikes
    warm_frame = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    for name, model in [
        ("exit", exit_model),
        ("walk_on", crosswalk_on_model),
        ("walk_off", crosswalk_off_model),
        ("crosswalk", crosswalk_model),
        ("school", school_sign_model),
    ]:
        if model is None:
            continue
        try:
            start = time.perf_counter()
            _ = run_predict(model, warm_frame)
            ms = (time.perf_counter() - start) * 1000
            print(f"[warmup] {name}: {ms:.1f}ms")
        except Exception as exc:
            print(f"[warmup] {name} failed: {exc}")


@app.get("/")
def root():
    return {"status": "good"}


@app.get("/health")
def health():
    models_loaded = {
        "exit_model": exit_model is not None,
        "crosswalk_on_model": crosswalk_on_model is not None,
        "crosswalk_off_model": crosswalk_off_model is not None,
        "crosswalk_model": crosswalk_model is not None,
        "school_sign_model": school_sign_model is not None,
    }
    return {"status": "ok", "models": models_loaded}


@app.post("/voice_input")
async def voice_input(audio: UploadFile = File(...)):
    if groq_client is None:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY missing")

    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="No audio provided")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as temp:
        temp.write(contents)
        temp_path = temp.name

    try:
        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio.filename or "audio.m4a", f),
                model="whisper-large-v3",
            )
        return {"text": transcription.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


@app.post("/signs")
def signs(image: UploadFile = File(...)):
    # sync endpoint: better for CPU-bound inference
    if all(
        model_obj is None
        for model_obj in [
            exit_model,
            crosswalk_on_model,
            crosswalk_off_model,
            crosswalk_model,
            school_sign_model,
        ]
    ):
        raise HTTPException(status_code=500, detail="No valid .pt model found in models folder")

    t0 = time.perf_counter()
    raw_image = image.file.read()
    frame = preprocess_frame(raw_image)

    detected_signs = empty_detected_signs()
    best_area_by_sign: Dict[str, float] = {}

    frame_height, frame_width = frame.shape[:2]

    # Exit model
    if exit_model is not None:
        t = time.perf_counter()
        exit_predictions = run_predict(exit_model, frame)
        print(f"[signs] exit_model: {(time.perf_counter() - t) * 1000:.1f}ms")

        exit_result = exit_predictions[0]
        for box in exit_result.boxes:
            class_id = int(box.cls[0])
            sign_key = id_to_sign.get(class_id)
            if sign_key in detected_signs:
                update_sign_with_box(
                    detected_signs, best_area_by_sign, sign_key, box, frame_width, frame_height
                )

    # One-class auxiliary models
    extra_model_targets: Tuple[Tuple[Optional[YOLO], str], ...] = (
        (crosswalk_on_model, "walk_on"),
        (crosswalk_off_model, "walk_off"),
        (crosswalk_model, "crosswalk"),
        (school_sign_model, "school_crosswalk"),
    )

    for model_obj, sign_key in extra_model_targets:
        if model_obj is None:
            continue

        t = time.perf_counter()
        prediction = run_predict(model_obj, frame)
        print(f"[signs] {sign_key}_model: {(time.perf_counter() - t) * 1000:.1f}ms")

        model_result = prediction[0]
        model_names = model_result.names

        for box in model_result.boxes:
            class_id = int(box.cls[0])
            class_name = model_names[class_id] if isinstance(model_names, (list, tuple)) else model_names[class_id]
            if normalize_sign_name(class_name) != sign_key:
                continue

            update_sign_with_box(
                detected_signs, best_area_by_sign, sign_key, box, frame_width, frame_height
            )

    total_ms = (time.perf_counter() - t0) * 1000
    print(f"[signs] total: {total_ms:.1f}ms")

    return detected_signs


@app.post("/api/route")
def get_route(route: RouteRequest):
    if not ORS_API_KEY:
        raise HTTPException(status_code=500, detail="ORS_API_KEY missing")

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
        response = http_session.post(url, json=body, headers=headers, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        detail = response.text if "response" in locals() else "ORS HTTP error"
        status_code = response.status_code if "response" in locals() else 502
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))