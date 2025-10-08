from fastapi.testclient import TestClient
from src.api.main import app

def test_contract():
    client = TestClient(app)
    # Parámetros dummy; retornará 503 si no hay silver
    r = client.get("/predict", params={"lat": -34.6, "lon": -58.4, "minutosLlegada": 10})
    assert r.status_code in (200, 503, 422)
