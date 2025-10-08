from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from haversine import haversine
from ..core.config import get_settings
from ..models.registry_loader import load_models, available_horizons, load_models_for_horizon
from pathlib import Path
import math

app = FastAPI(title="HayBici API", version="1.0.0")
settings = get_settings()

# Cargamos modelos si existen
MODELS_READY = (Path("models/registry/regressor.joblib").exists() and
                Path("models/registry/classifier_calibrated.joblib").exists() and
                Path("models/registry/meta.joblib").exists())
if MODELS_READY:
    REG, CAL, META = load_models()
    FEATS = META["features"]
    FEATURES_VERSION = META.get("features", [])
    FEATURES_VERSION = META.get("features_version", "f1.0.0")
else:
    REG = CAL = META = None
    FEATS = []
    FEATURES_VERSION = "f0.0.0"

class PredictionItem(BaseModel):
    station_id: str
    distance_m: float
    y_hat: float
    p_availability: float
    prediction_ts: str
    horizon_min: int
    features_version: str
    model_version: str

def now_local():
    return datetime.now(tz=pytz.timezone(settings.yaml_cfg["time"]["local_tz"]))

def parse_target_ts(horaLlegada: Optional[str], minutosLlegada: Optional[int]):
    if horaLlegada:
        try:
            hh, mm = map(int, horaLlegada.split(":"))
            n = now_local()
            target = n.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target < n:
                raise HTTPException(422, "horaLlegada debe ser futura (mismo día).")
            horizon = int((target - n).total_seconds() // 60)
            return target, horizon
        except Exception:
            raise HTTPException(422, "Formato de horaLlegada inválido. Use HH:mm")
    if minutosLlegada is not None:
        if minutosLlegada < 0 or minutosLlegada > settings.yaml_cfg["time"]["max_horizon_min"]:
            raise HTTPException(422, "minutosLlegada fuera de rango.")
        n = now_local()
        target = n + timedelta(minutes=int(minutosLlegada))
        return target, int(minutosLlegada)
    raise HTTPException(422, "Debe indicar horaLlegada o minutosLlegada.")

def load_current_status():
    # Usamos silver/status_clean.parquet como estado base
    pq = Path(settings.yaml_cfg["data"]["base_path"]) / "silver" / "status_clean.parquet"
    if not pq.exists():
        raise HTTPException(503, "No hay datos en silver. Ejecuta scripts/make_dataset.py")
    df = pd.read_parquet(pq)
    df.columns = [c.lower() for c in df.columns]
    # último snapshot por estación
    df = df.sort_values(["station_id","ts_local"]).groupby("station_id").tail(1).reset_index(drop=True)
    return df

def build_point_features(df: pd.DataFrame, target_ts: datetime) -> pd.DataFrame:
    # Para demo v1: usamos las columnas ya presentes y rellenamos lags con el último valor disponible (simple).
    # El pipeline real debería precomputar features minutely en gold.
    df = df.copy()
    # Señales de calendario del target
    hour = target_ts.hour + target_ts.minute/60.0
    df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
    df["hour_cos"] = np.cos(2*np.pi*hour/24.0)
    dow = target_ts.weekday()
    for d in range(7):
        df[f"dow_{d}"] = 1 if d == dow else 0
    # Placeholders si el modelo espera otros features: los seteamos a NaN y dejamos que scikit los maneje si corresponde
    for f in FEATS:
        if f not in df.columns:
            df[f] = np.nan
    return df

def topk_by_distance(df: pd.DataFrame, lat: float, lon: float, k: int) -> pd.DataFrame:
    def dist(row):
        return haversine((lat, lon), (row["lat"], row["lon"])) * 1000.0
    df["distance_m"] = df.apply(dist, axis=1)
    return df.sort_values("distance_m").head(k).copy()

def nearest_horizon(requested_min: int, trained: list[int]) -> int:
    if not trained:
        return requested_min
    # clamp al rango entrenado
    lo, hi = min(trained), max(trained)
    req = max(lo, min(hi, requested_min))
    # elegir el más cercano
    best = min(trained, key=lambda h: abs(h - req))
    return best


@app.get("/predict", response_model=List[PredictionItem])
def predict(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    horaLlegada: Optional[str] = None,
    minutosLlegada: Optional[int] = None,
    topN: int = Query(None, ge=1, le=10),
):
    cfg = settings.yaml_cfg
    if topN is None:
        topN = cfg["api"]["default_topN"]
    target_ts, horizon = parse_target_ts(horaLlegada, minutosLlegada)
    if horizon > cfg["time"]["max_horizon_min"]:
        raise HTTPException(422, "Horizonte excede el máximo configurado.")

    state = load_current_status()
    # Validaciones mínimas
    for col in ["lat", "lon", "num_bikes_available", "station_id"]:
        if col not in state.columns:
            raise HTTPException(503, f"Columna requerida faltante en silver: {col}")

    kdf = topk_by_distance(state, lat, lon, k=topN)
    feats = build_point_features(kdf, target_ts)

    trained_H = available_horizons()
    chosen_H = nearest_horizon(horizon, trained_H) if trained_H else horizon  # si no hay modelos, usamos fallback

    results = []
    if trained_H:
        # cargar modelos y features del H elegido
        REG, CAL, FEATS, FEATURES_VERSION = load_models_for_horizon(chosen_H)
        model_version = f"m_gbdt_h{chosen_H}"
    else:
        REG = CAL = None
        FEATS = []
        FEATURES_VERSION = "f0.0.0"
        model_version = "m0.0.0"

    for _, row in feats.iterrows():
        if REG is not None:
            X = row.reindex(FEATS, fill_value=np.nan).to_frame().T.astype(float)
            y_hat = float(REG.predict(X)[0])
            # CAL puede ser clasificador con predict_proba
            if hasattr(CAL, "predict_proba"):
                p_av = float(CAL.predict_proba(X)[:,1][0])
            else:
                # fallback si fuera un clasificador sin predict_proba
                s = CAL.decision_function(X)
                p_av = float((s - s.min()) / (s.max() - s.min() + 1e-9))
        else:
            y_hat = float(max(row.get("num_bikes_available", 0.0), 0.0))
            p_av = 1.0 if y_hat >= 1 else 0.0

        results.append({
            "station_id": str(row["station_id"]),
            "distance_m": float(row["distance_m"]),
            "y_hat": y_hat,
            "p_availability": p_av,
            "prediction_ts": target_ts.isoformat(),
            "horizon_min": horizon,             # lo que pidió el user
            "features_version": FEATURES_VERSION,
            "model_version": model_version      # indica H del modelo usado
        })


    # Ranking ponderado por distancia y probabilidad
    alpha = cfg["ranking"]["alpha_distance"]
    for r in results:
        w_dist = math.exp(-alpha * r["distance_m"])
        # normalizar y_hat aproximado (capacidad desconocida -> usar 10 como escala blandita)
        y_norm = min(r["y_hat"]/10.0, 1.0)
        r["_score"] = 0.5*w_dist + 0.5*max(r["p_availability"], y_norm)
    results = sorted(results, key=lambda x: x["_score"], reverse=True)
    for r in results: r.pop("_score", None)
    return results
