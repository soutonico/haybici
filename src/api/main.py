from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from haversine import haversine
from ..core.config import get_settings
from ..models.registry_loader import available_horizons, load_models_for_horizon
from pathlib import Path
import math

app = FastAPI(title="HayBici API", version="1.0.0")
settings = get_settings()

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

def build_point_features(kdf: pd.DataFrame, target_ts: pd.Timestamp, feat_names: Optional[list] = None) -> pd.DataFrame:
    """
    Construye features para predicción en target_ts para las estaciones de kdf.
    NOTA: Por ahora, si faltan lags/rolling, se crean como 0.0 para que el modelo pueda predecir.
          (v1 simple; luego leeremos histórico de silver para calcularlos de verdad)
    """
    df = kdf.copy()

    # Hora local -> hour_sin/cos y one-hot de día (dow_*)
    ts = pd.to_datetime(target_ts)
    hour = ts.hour + ts.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ts.weekday()
    for d in range(7):
        df[f"dow_{d}"] = 1.0 if d == dow else 0.0

    # capacity puede venir faltante: dejar como está; el modelo tolera 0.0 si reindexeamos luego
    if "capacity" not in df.columns:
        df["capacity"] = 0.0

    # Asegurar columnas esperadas por el modelo (si nos las pasan)
    if feat_names:
        for f in feat_names:
            if f not in df.columns:
                # v1: rellenamos con 0.0 (lags/rolling ausentes)
                df[f] = 0.0

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

    # 1) Timestamp objetivo y horizonte
    target_ts, horizon = parse_target_ts(horaLlegada, minutosLlegada)
    if horizon > cfg["time"]["max_horizon_min"]:
        raise HTTPException(422, "Horizonte excede el máximo configurado.")

    # 2) Estado actual (silver) y validaciones mínimas
    state = load_current_status()
    for col in ["lat", "lon", "num_bikes_available", "station_id"]:
        if col not in state.columns:
            raise HTTPException(503, f"Columna requerida faltante en silver: {col}")

    # 3) Elegir modelo por horizonte entrenado más cercano (ANTES de armar features)
    trained_H = available_horizons()
    chosen_H = nearest_horizon(horizon, trained_H) if trained_H else horizon
    family = (cfg.get("model", {}).get("family") or "sklearn_gbdt")

    if trained_H:
        REG, CAL, FEAT_NAMES, FEATURES_VERSION = load_models_for_horizon(chosen_H)
        model_version = f"m_{'lgbm' if family=='lgbm' else 'gbdt'}_h{chosen_H}"
    else:
        REG = CAL = None
        FEAT_NAMES = []
        FEATURES_VERSION = "f0.0.0"
        model_version = "m0.0.0"

    # 4) Top-N por distancia y features para ese instante futuro (usando feat_names)
    kdf = topk_by_distance(state, lat, lon, k=topN)
    feats = build_point_features(kdf, target_ts, feat_names=FEAT_NAMES)   # ← cambiamos firma

    # 5) Predecir por estación
    results = []
    for _, row in feats.iterrows():
        X = row.reindex(FEAT_NAMES, fill_value=0.0).to_frame().T.astype(float) if FEAT_NAMES else None

        if REG is not None and X is not None:
            y_hat = float(REG.predict(X)[0])
            if hasattr(CAL, "predict_proba"):
                p_av = float(CAL.predict_proba(X)[:, 1][0])
            else:
                s = CAL.decision_function(X)
                p_av = float((s - s.min()) / (s.max() - s.min() + 1e-9))
        else:
            # Fallback sin modelo
            y_hat = float(max(row.get("num_bikes_available", 0.0), 0.0))
            p_av = 1.0 if y_hat >= 1 else 0.0

        results.append({
            "station_id": str(row["station_id"]),
            "distance_m": float(row["distance_m"]),
            "y_hat": y_hat,
            "p_availability": p_av,
            "prediction_ts": target_ts.isoformat(),
            "horizon_min": int(horizon),
            "features_version": FEATURES_VERSION,
            "model_version": model_version,
            # "horizon_model_min": int(chosen_H),
        })

    # 6) Ranking final
    alpha = cfg["ranking"]["alpha_distance"]
    for r in results:
        w_dist = math.exp(-alpha * r["distance_m"])
        y_norm = min(r["y_hat"] / 10.0, 1.0)  # normalización blanda
        r["_score"] = 0.5 * w_dist + 0.5 * max(r["p_availability"], y_norm)
    results.sort(key=lambda x: x["_score"], reverse=True)
    for r in results:
        r.pop("_score", None)

    return results[:topN]
