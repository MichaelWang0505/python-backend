import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

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
# Image processing config
# --------------------------
MAX_IMAGE_SIDE = 960
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_IOU = 0.45

# --------------------------
# Camera config
# iPhone 14 Plus wide camera: ~26mm equivalent focal length
# Sensor width ~6.3mm, image width at MAX_IMAGE_SIDE=960
# Pixel focal length = (focal_mm / sensor_width_mm) * image_width_px
# --------------------------
FOCAL_LENGTH_MM = 26.0        # iPhone 14 Plus wide lens equivalent focal length (mm)
SENSOR_WIDTH_MM = 6.3         # iPhone 14 Plus sensor width (mm)
IMAGE_WIDTH_PX  = 960         # Max image side after preprocessing

FOCAL_LENGTH_PX = (FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * IMAGE_WIDTH_PX

# --------------------------
# Real-world sign dimensions (meters)
# These are the dimensions of the BOUNDING BOX region that was labeled,
# not the full physical sign. Update these if you find better values.
# --------------------------

# Exit sign — bounding box around "EXIT" text only (not full sign with white border)
EXIT_SIGN_WIDTH_M  = 0.30   # 12 inches
EXIT_SIGN_HEIGHT_M = 0.08   # 3 inches

# Walk / Don't walk — bounding box around white person or red hand symbol only
WALK_SIGN_WIDTH_M  = 0.15   # 6 inches
WALK_SIGN_HEIGHT_M = 0.15   # 6 inches

# Pedestrian crossing — full diamond sign bounding box
CROSSWALK_SIGN_WIDTH_M  = 0.76  # 30 inches
CROSSWALK_SIGN_HEIGHT_M = 0.76  # 30 inches

# School crossing — full diamond sign bounding box
SCHOOL_SIGN_WIDTH_M  = 0.76  # 30 inches
SCHOOL_SIGN_HEIGHT_M = 0.76  # 30 inches

# --------------------------
# Model directory
# --------------------------
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


# Load all models
exit_model      = load_model("exit_signs.pt")
walk_model      = load_model("walk_sign.pt")
crosswalk_model = load_model("crosswalk.pt")
school_model    = load_model("school_sign.pt")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Exit sign class mapping
exit_id_to_sign = {
    0: "exit_sign",
    1: "exit_right",
    2: "exit_left",
    3: "exit_both_ways",
}

# Walk sign class mapping (0=on, 1=off)
walk_id_to_sign = {
    0: "walk_on",
    1: "walk_off",
}


def empty_detected_signs() -> Dict[str, Dict[str, object]]:
    return {
        "exit_sign":           {"detected": False, "direction": "center", "distance": 0},
        "exit_right":          {"detected": False, "direction": "center", "distance": 0},
        "exit_left":           {"detected": False, "direction": "center", "distance": 0},
        "exit_both_ways":      {"detected": False, "direction": "center", "distance": 0},
        "walk_on":             {"detected": False, "direction": "center", "distance": 0},
        "walk_off":            {"detected": False, "direction": "center", "distance": 0},
        "crosswalk":           {"detected": False, "direction": "center", "distance": 0},
        "school_crosswalk":    {"detected": False, "direction": "center", "distance": 0},
    }


def preprocess_frame(raw_image: bytes) -> np.ndarray:
    if not raw_image:
        raise HTTPException(status_code=400, detail="No image")

    image_arr = np.frombuffer(raw_image, np.uint8)
    frame = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    h, w = frame.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max(h, w))
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

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


def estimate_distance(
    box_width_px: float,
    box_height_px: float,
    real_width_m: float,
    real_height_m: float,
) -> float:
    """
    Estimate distance in meters using the pinhole camera model:
        distance = (real_size_m * focal_length_px) / box_size_px

    We compute distance from both width and height independently,
    then take the minimum (closer estimate is more reliable since
    the bounding box may not be perfectly tight on both axes).
    """
    dist_from_width  = (real_width_m  * FOCAL_LENGTH_PX) / box_width_px  if box_width_px  > 0 else float("inf")
    dist_from_height = (real_height_m * FOCAL_LENGTH_PX) / box_height_px if box_height_px > 0 else float("inf")
    return min(dist_from_width, dist_from_height)


def run_predict(model: YOLO, frame: np.ndarray):
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
    best_distance_by_sign: Dict[str, float],
    sign_key: str,
    box,
    frame_width: int,
    frame_height: int,
    real_width_m: float,
    real_height_m: float,
) -> None:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    box_width_px  = max(0.0, x2 - x1)
    box_height_px = max(0.0, y2 - y1)

    distance_m = estimate_distance(box_width_px, box_height_px, real_width_m, real_height_m)

    # Keep the detection with the smallest (closest) distance
    if sign_key not in best_distance_by_sign or distance_m < best_distance_by_sign[sign_key]:
        x_center = (x1 + x2) / 2.0

        detected_signs[sign_key]["detected"]  = True
        detected_signs[sign_key]["direction"] = classify_direction(x_center, frame_width)
        detected_signs[sign_key]["distance"]  = round(distance_m, 2)

        best_distance_by_sign[sign_key] = distance_m


@app.on_event("startup")
def warmup_models():
    warm_frame = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
    for name, model in [
        ("exit",      exit_model),
        ("walk",      walk_model),
        ("crosswalk", crosswalk_model),
        ("school",    school_model),
    ]:
        if model is None:
            print(f"[warmup] {name} model not loaded, skipping")
            continue
        try:
            start = time.perf_counter()
            run_predict(model, warm_frame)
            print(f"[warmup] {name}: {(time.perf_counter() - start) * 1000:.1f}ms")
        except Exception as exc:
            print(f"[warmup] {name} failed: {exc}")


@app.get("/")
def root():
    return {"status": "good"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "exit_model":      exit_model      is not None,
            "walk_model":      walk_model      is not None,
            "crosswalk_model": crosswalk_model is not None,
            "school_model":    school_model    is not None,
        }
    }


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
    t0 = time.perf_counter()
    raw_image = image.file.read()
    frame = preprocess_frame(raw_image)

    detected_signs = empty_detected_signs()
    best_distance_by_sign: Dict[str, float] = {}
    frame_height, frame_width = frame.shape[:2]

    # --- Exit signs ---
    if exit_model is not None:
        t = time.perf_counter()
        exit_result = run_predict(exit_model, frame)[0]
        print(f"[signs] exit_model: {(time.perf_counter() - t) * 1000:.1f}ms")
        for box in exit_result.boxes:
            class_id = int(box.cls[0])
            sign_key = exit_id_to_sign.get(class_id)
            if sign_key in detected_signs:
                update_sign_with_box(
                    detected_signs, best_distance_by_sign, sign_key, box,
                    frame_width, frame_height,
                    EXIT_SIGN_WIDTH_M, EXIT_SIGN_HEIGHT_M,
                )

    # --- Walk / Don't walk signs ---
    if walk_model is not None:
        t = time.perf_counter()
        walk_result = run_predict(walk_model, frame)[0]
        print(f"[signs] walk_model: {(time.perf_counter() - t) * 1000:.1f}ms")
        for box in walk_result.boxes:
            class_id = int(box.cls[0])
            sign_key = walk_id_to_sign.get(class_id)
            if sign_key in detected_signs:
                update_sign_with_box(
                    detected_signs, best_distance_by_sign, sign_key, box,
                    frame_width, frame_height,
                    WALK_SIGN_WIDTH_M, WALK_SIGN_HEIGHT_M,
                )

    # --- Crosswalk signs ---
    if crosswalk_model is not None:
        t = time.perf_counter()
        crosswalk_result = run_predict(crosswalk_model, frame)[0]
        print(f"[signs] crosswalk_model: {(time.perf_counter() - t) * 1000:.1f}ms")
        for box in crosswalk_result.boxes:
            update_sign_with_box(
                detected_signs, best_distance_by_sign, "crosswalk", box,
                frame_width, frame_height,
                CROSSWALK_SIGN_WIDTH_M, CROSSWALK_SIGN_HEIGHT_M,
            )

    # --- School crossing signs ---
    if school_model is not None:
        t = time.perf_counter()
        school_result = run_predict(school_model, frame)[0]
        print(f"[signs] school_model: {(time.perf_counter() - t) * 1000:.1f}ms")
        for box in school_result.boxes:
            update_sign_with_box(
                detected_signs, best_distance_by_sign, "school_crosswalk", box,
                frame_width, frame_height,
                SCHOOL_SIGN_WIDTH_M, SCHOOL_SIGN_HEIGHT_M,
            )

    print(f"[signs] total: {(time.perf_counter() - t0) * 1000:.1f}ms")
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