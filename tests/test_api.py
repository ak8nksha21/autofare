from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID = {
    "distance_km": 10.0, "wait_min": 5.0, "passengers": 2, "hour": 14,
    "luggage_bags": 1, "fuel_price_rs": 100.0, "booking": "app", "zone": "Market",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


def test_fuel_price_above_150_rejected():
    r = client.post("/predict", json={**VALID, "fuel_price_rs": 151})
    assert r.status_code == 422


def test_negative_wait_rejected():
    r = client.post("/predict", json={**VALID, "wait_min": -1})
    assert r.status_code == 422


def test_missing_field_rejected():
    bad = {k: v for k, v in VALID.items() if k != "zone"}
    assert client.post("/predict", json=bad).status_code == 422
