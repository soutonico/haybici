import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pytz
import yaml
import json
from typing import List, Optional

CONFIG_PATH = Path("config/config.yaml")

def load_cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs(base_path: Path, layers: dict):
    for lvl in layers.values():
        (base_path / lvl).mkdir(parents=True, exist_ok=True)
        # opcional: subcarpeta raw para guardar JSON si querés
        if lvl == "bronze":
            (base_path / lvl / "raw").mkdir(parents=True, exist_ok=True)

def _read_many_json(paths: List[Path]) -> List[dict]:
    out = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as e:
            print(f"[WARN] No pude leer {p}: {e}")
    return out

def _gbfs_to_df(docs: List[dict], key: str) -> pd.DataFrame:
    # key = "station_information" o "station_status": esperamos docs[i]["data"]["stations"]
    rows = []
    for d in docs:
        stations = d.get("data", {}).get("stations", [])
        last_updated = d.get("last_updated", None)  # epoch sec en GBFS
        for s in stations:
            r = dict(s)
            r["_file_last_updated"] = last_updated
            rows.append(r)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def _to_local_from_epoch(col: pd.Series, local_tz: str) -> pd.Series:
    """Convierte epoch (sec o ms) -> tz local; ignora NaT."""
    if col.isna().all():
        return pd.to_datetime(col)
    # detectar sec vs ms
    # si hay valores grandes (~ms) los pasamos como ms, si no, como sec
    series = col.astype("float64")
    use_ms = (series.dropna() > 1e12).mean() > 0.5
    unit = "ms" if use_ms else "s"
    s = pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
    return s.dt.tz_convert(pytz.timezone(local_tz))

def _flatten_rental_uris(df: pd.DataFrame) -> pd.DataFrame:
    if "rental_uris" in df.columns:
        # Crear columnas planas
        def get_android(x):
            return x.get("android") if isinstance(x, dict) else None
        def get_ios(x):
            return x.get("ios") if isinstance(x, dict) else None

        df["rental_uri_android"] = df["rental_uris"].apply(get_android)
        df["rental_uri_ios"] = df["rental_uris"].apply(get_ios)
        # Quitar la columna conflictiva
        df = df.drop(columns=["rental_uris"])
    return df

def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evita tipos no soportados por Parquet (struct vacío, listas/dicts heterogéneos).
    - Aplana rental_uris.
    - Cualquier dict/list restante -> string JSON (preserva info sin romper).
    """
    df = _flatten_rental_uris(df)
    # Detectar columnas con dict/list
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        # Si hay algún dict o list en la columna, lo pasamos a JSON string
        if df[c].apply(lambda x: isinstance(x, (dict, list))).any():
            df[c] = df[c].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)
    return df


def build_bronze_from_json(
    info_glob: Optional[str],
    status_glob: Optional[str],
    out_dir: Path,
    local_tz: str,
) -> None:
    info_docs = _read_many_json(sorted(Path().glob(info_glob))) if info_glob else []
    status_docs = _read_many_json(sorted(Path().glob(status_glob))) if status_glob else []

    info_df = _gbfs_to_df(info_docs, "station_information") if info_docs else pd.DataFrame()
    stat_df = _gbfs_to_df(status_docs, "station_status") if status_docs else pd.DataFrame()

    if not info_df.empty:
        info_df.columns = [c.lower() for c in info_df.columns]
        # >>> PATCH: sanitizar antes de parquet
        info_df = _sanitize_for_parquet(info_df)
        (out_dir / "bronze").mkdir(parents=True, exist_ok=True)
        info_df.to_parquet(out_dir / "bronze" / "station_information.parquet", index=False)
        print(f"bronze/information → {len(info_df)} filas")

    if not stat_df.empty:
        stat_df.columns = [c.lower() for c in stat_df.columns]
        # >>> PATCH: sanitizar antes de parquet (por si trae listas/dicts)
        stat_df = _sanitize_for_parquet(stat_df)
        (out_dir / "bronze").mkdir(parents=True, exist_ok=True)
        stat_df.to_parquet(out_dir / "bronze" / "station_status.parquet", index=False)
        print(f"bronze/status → {len(stat_df)} filas")


def build_bronze_from_parquet(status_parquet: Path, out_dir: Path) -> None:
    df = pd.read_parquet(status_parquet)
    df.columns = [c.lower() for c in df.columns]
    (out_dir / "bronze").mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "bronze" / "station_status.parquet", index=False)
    print(f"bronze/status (desde parquet unido) → {len(df)} filas")

def build_silver(base_path: Path, local_tz: str) -> None:
    b = base_path / "bronze"
    info_pq = b / "station_information.parquet"
    stat_pq = b / "station_status.parquet"
    assert stat_pq.exists(), "No hay bronze/station_status.parquet"

    stat = pd.read_parquet(stat_pq)
    stat.columns = [c.lower() for c in stat.columns]

    # --- Fallback robusto para timestamp ---
    # Preferimos 'last_reported'. Si falta o es NaN, usamos '_file_last_updated' (del JSON)
    cand_cols = []
    if "last_reported" in stat.columns:
        cand_cols.append("last_reported")
    if "_file_last_updated" in stat.columns:
        cand_cols.append("_file_last_updated")

    ts_local = None
    if cand_cols:
        # Tomamos la primera columna disponible con no-nulos por fila
        # Creamos una serie combinada
        import numpy as np
        base = stat[cand_cols[0]]
        for c in cand_cols[1:]:
            base = base.fillna(stat[c])
        ts_local = _to_local_from_epoch(base, local_tz)

    stat["ts_local"] = ts_local
    # Quitamos filas sin timestamp utilizable
    stat = stat[stat["ts_local"].notna()].copy()

    # Join con information si existe (para lat/lon/capacity/name)
    if info_pq.exists():
        info = pd.read_parquet(info_pq)
        info.columns = [c.lower() for c in info.columns]
        join_cols = [c for c in ["station_id", "name", "lat", "lon", "capacity", "address"] if c in info.columns]
        if "_file_last_updated" in info.columns:
            info = info.sort_values(["station_id", "_file_last_updated"]).groupby("station_id").tail(1)
        stat = stat.merge(info[join_cols], on="station_id", how="left")

    # Flags y filtro
    for col in ["is_renting", "is_returning"]:
        if col not in stat.columns:
            stat[col] = 1
    stat["is_closed"] = (stat["is_renting"] == 0) | (stat["is_returning"] == 0)
    stat = stat[~stat["is_closed"]].copy()

    # Columnas mínimas
    for c in ["station_id", "num_bikes_available", "lat", "lon", "capacity", "name"]:
        if c not in stat.columns:
            stat[c] = np.nan

    stat = stat.sort_values(["station_id", "ts_local"])
    out_pq = base_path / "silver" / "status_clean.parquet"
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    stat.to_parquet(out_pq, index=False)
    print(f"silver/status_clean → {len(stat)} filas | estaciones={stat['station_id'].nunique()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-json-glob", type=str, help="Glob a stationInformation_*.json (p.ej. data/bronze/raw/stationInformation_*.json)")
    parser.add_argument("--status-json-glob", type=str, help="Glob a stationStatus_*.json (p.ej. data/bronze/raw/stationStatus_*.json)")
    parser.add_argument("--status-parquet", type=str, help="Ruta a ecobici_station_status.parquet (unido)")
    args = parser.parse_args()

    cfg = load_cfg()
    base_path = Path(cfg["data"]["base_path"])
    ensure_dirs(base_path, cfg["data"]["layers"])
    local_tz = cfg["time"]["local_tz"]

    # 1) Construir bronze con lo disponible
    if args.info_json_glob or args.status_json_glob:
        build_bronze_from_json(args.info_json_glob, args.status_json_glob, base_path, local_tz)

    if args.status_parquet:
        build_bronze_from_parquet(Path(args.status_parquet), base_path)

    # 2) Silver
    build_silver(base_path, local_tz)

if __name__ == "__main__":
    main()
