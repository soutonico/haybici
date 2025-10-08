import json
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

CONFIG_PATH = Path("config/config.yaml")
OUT_JSON = Path("models/registry/hour_weights.json")

def load_cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    silver = Path(cfg["data"]["base_path"]) / "silver" / "status_clean.parquet"
    if not silver.exists():
        raise FileNotFoundError(f"No existe {silver}. Corré primero: scripts/make_dataset.py")

    df = pd.read_parquet(silver)
    df.columns = [c.lower() for c in df.columns]

    # --- Parámetros ---
    noise_threshold = float(cfg.get("weights", {}).get("noise_threshold", 0.5))
    cap_by_capacity = bool(cfg.get("weights", {}).get("cap_by_capacity", True))
    max_delta_per_min = float(cfg.get("weights", {}).get("max_delta_per_min", 30))
    min_w = float(cfg.get("weights", {}).get("min_weight", 0.5))
    max_w = float(cfg.get("weights", {}).get("max_weight", 1.5))

    # --- Pre: si hay múltiples snapshots en el mismo minuto, tomar el último ---
    df["ts_local"] = pd.to_datetime(df["ts_local"], errors="coerce")
    df = df[df["ts_local"].notna()].copy()
    df["minute"] = df["ts_local"].dt.floor("min")
    df = (df.sort_values(["station_id", "ts_local"])
            .groupby(["station_id", "minute"])
            .tail(1)
            .drop(columns="minute"))

    # --- Delta por estación (1 min) ---
    df = df.sort_values(["station_id", "ts_local"])
    if "num_bikes_available" not in df.columns:
        raise ValueError("Falta columna num_bikes_available en silver.")
    df["delta_bikes"] = df.groupby("station_id")["num_bikes_available"].diff()

    # (1) Ignorar micro-ruido: solo consideramos caídas significativas
    #     Si delta >= -noise_threshold → salida estimada = 0
    dep = np.where(df["delta_bikes"] < -noise_threshold, -df["delta_bikes"], 0.0)

    # (3) Descartar outliers imposibles (> max_delta_per_min)
    dep = np.where(dep > max_delta_per_min, 0.0, dep)

    # (2) Cap por capacidad (si está disponible)
    if cap_by_capacity and "capacity" in df.columns:
        cap_series = df.groupby("station_id")["capacity"].ffill().bfill()
        dep_series = pd.Series(dep, index=df.index, dtype=float)
        cap_array = cap_series.fillna(np.inf).astype(float).to_numpy()
        dep = np.minimum(dep_series.to_numpy(), cap_array)

    df["dep_estimada"] = dep

    # --- Agregación por hora local ---
    df["hora"] = df["ts_local"].dt.hour
    uso = df.groupby("hora")["dep_estimada"].sum()

    # --- Normalización de pesos a [min_w, max_w] ---
    if uso.empty or uso.sum() == 0:
        weights = {str(h): 1.0 for h in range(24)}
    else:
        # escalar lineal a [min_w, max_w]
        umin, umax = float(uso.min()), float(uso.max())
        if umax == umin:
            weights = {str(h): 1.0 for h in range(24)}
        else:
            span = max_w - min_w
            norm = (uso - umin) / (umax - umin)  # 0..1
            scaled = min_w + norm * span
            weights = {str(int(h)): float(scaled.loc[h]) for h in scaled.index}
        # completar horas faltantes
        for h in range(24):
            weights.setdefault(str(h), min_w)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(weights, indent=2, ensure_ascii=False))
    print(f"✅ Pesos por hora guardados en {OUT_JSON}")
    print(weights)

if __name__ == "__main__":
    main()
