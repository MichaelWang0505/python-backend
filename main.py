from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/voice_input")
def voice_input():
    return {"text": ""}

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
    