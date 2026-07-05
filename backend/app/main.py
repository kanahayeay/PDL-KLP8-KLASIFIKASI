from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.model_loader import load_model_bundle
from app.preprocessing import preprocess
from app.schemas import PredictRequest, PredictResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_bundle = None


@app.on_event("startup")
def startup_event():
    global model_bundle
    model_bundle = load_model_bundle(config.ARTIFACTS_DIR)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        return {
            "hate_speech": {"label": 0, "probability": 0.0},
            "abusive": {"label": 0, "probability": 0.0},
        }
    tokens = preprocess(req.text)
    if not tokens:
        return {
            "hate_speech": {"label": 0, "probability": 0.0},
            "abusive": {"label": 0, "probability": 0.0},
        }
    return model_bundle.predict(tokens)