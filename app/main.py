import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

# Same column order as training (numbers first, then text columns).
FEATURES = ["distance_km", "wait_min", "passengers", "hour",
            "luggage_bags", "fuel_price_rs", "booking", "zone"]


class JsonFormatter(logging.Formatter):
    """Format every log record as one JSON line."""

    def format(self, record):
        payload = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_data", {}))
        return json.dumps(payload)


logger = logging.getLogger("autofare.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)  # screen only, never a file
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


class RideRequest(BaseModel):
    distance_km: float
    wait_min: float = Field(..., ge=0)            # client rule 2: below 0 is rejected
    passengers: int
    hour: int
    luggage_bags: int
    fuel_price_rs: float = Field(..., le=150)     # client rule 1: above 150 is rejected
    booking: str
    zone: str


app = FastAPI(title="Auto-Rickshaw Fare API")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: RideRequest):
    start = time.time()
    row = req.model_dump()
    row["booking"] = row["booking"].strip().lower()   # tidy spelling like in training
    frame = pd.DataFrame([row])[FEATURES]
    prediction = round(float(model.predict(frame)[0]), 2)

    logger.info("prediction", extra={"extra_data": {
        "features": row,
        "prediction": prediction,
        "latency_ms": round((time.time() - start) * 1000, 2),
    }})
    return {"prediction": prediction}
