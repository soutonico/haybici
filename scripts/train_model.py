import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import pytz
from sklearn.metrics import mean_absolute_error, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import joblib
import mlflow
import json
import sklearn
import json as _json


CONFIG_PATH = Path("config/config.yaml")
REG_ROOT = Path("models/registry")
REG_ROOT.mkdir(parents=True, exist_ok=True)

def load_cfg():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
    

def make_models(cfg):
    """
    Devuelve (regressor, classifier_base, family_str)
    según cfg["model"]["family"] y sus hiperparámetros.
    """
    family = (cfg.get("model", {}).get("family") or "sklearn_gbdt").lower()

    if family == "lgbm":
        try:
            from lightgbm import LGBMRegressor, LGBMClassifier
        except Exception as e:
            raise RuntimeError(
                "Configura family='lgbm' pero 'lightgbm' no está instalado. "
                "Instala con: pip install lightgbm OR cambia model.family a 'sklearn_gbdt'."
            ) from e

        rp = cfg["model"].get("lgbm", {}).get("regressor", {}) if "model" in cfg else {}
        cp = cfg["model"].get("lgbm", {}).get("classifier", {}) if "model" in cfg else {}
        reg = LGBMRegressor(**{
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            **rp
        })
        cls = LGBMClassifier(**{
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            **cp
        })
        return reg, cls, "lgbm"

    # Default: sklearn Gradient Boosting
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    rp = cfg["model"].get("sklearn_gbdt", {}).get("regressor", {}) if "model" in cfg else {}
    cp = cfg["model"].get("sklearn_gbdt", {}).get("classifier", {}) if "model" in cfg else {}
    reg = GradientBoostingRegressor(**{"random_state": 42, **rp})
    cls = GradientBoostingClassifier(**{"random_state": 42, **cp})
    return reg, cls, "sklearn_gbdt"

# ===== Feature builder =====
def build_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.sort_values(["station_id", "ts_local"]).copy()
    target = cfg["training"]["target_reg"]

    # asegurar tipo numérico en target
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # LAGS por estación
    for lag in cfg["features"]["lags_min"]:
        df[f"lag_{lag}"] = df.groupby("station_id")[target].shift(lag)

    # ROLLING por estación (mean/std) evitando el leak de t
    for w in cfg["features"]["rolling_windows_min"]:
        s = df.groupby("station_id")[target].shift(1)
        df[f"rollmean_{w}"] = s.groupby(df["station_id"]).rolling(w).mean().reset_index(level=0, drop=True)
        df[f"rollstd_{w}"]  = s.groupby(df["station_id"]).rolling(w).std().reset_index(level=0, drop=True)

    # Calendario
    if cfg["features"]["add_weekly_seasonality"]:
        df["dow"] = df["ts_local"].dt.weekday
        df = pd.get_dummies(df, columns=["dow"], drop_first=False, prefix="dow")
    if cfg["features"]["add_hourly_cyclical"]:
        hour = df["ts_local"].dt.hour + df["ts_local"].dt.minute/60.0
        df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
        df["hour_cos"] = np.cos(2*np.pi*hour/24.0)

    # Solo dropear NA en columnas que usará el modelo
    model_cols = [cfg["training"]["target_reg"]] + [
        c for c in df.columns
        if c.startswith(("lag_", "rollmean_", "rollstd_", "dow_", "hour_"))
    ]
    keep = df.dropna(subset=model_cols).copy()

    # Rellenar lat/lon/capacity por estación si hace falta (para API)
    for col in ["lat","lon","capacity"]:
        if col in keep.columns:
            keep[col] = keep.groupby("station_id")[col].ffill().bfill()

    return keep.reset_index(drop=True)

def pick_feats(df: pd.DataFrame):
    # Solo engineered + numéricas
    engineered = [c for c in df.columns if c.startswith(("lag_", "rollmean_", "rollstd_", "dow_", "hour_"))]
    candidatas = engineered + ([c for c in ["capacity"] if c in df.columns])
    feats = []
    for c in candidatas:
        if np.issubdtype(pd.Series(df[c]).dtype, np.number):
            feats.append(c)
        else:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().any():
                df[c] = conv
                feats.append(c)
    df[feats] = df[feats].astype(float)
    return df, feats

def ensure_datetime_local(df):
    df["ts_local"] = pd.to_datetime(df["ts_local"], errors="coerce")
    df = df[df["ts_local"].notna()].copy()
    return df.sort_values(["station_id","ts_local"]).reset_index(drop=True)


def train_one_horizon(train_df, test_df, feats, H, cfg, sample_w=None, mlflow_active=False):
    """
    Entrena y evalúa un horizonte H (min). No imprime logs por pantalla.
    Devuelve un dict con métricas y paths, y persiste artefactos en models/registry/h{H}/.
    """
    target_reg = cfg["training"]["target_reg"]

    # ===== Targets a futuro (t+H) =====
    def add_future_targets(df_in):
        df_out = df_in.copy()
        df_out[f"y_reg_{H}"] = df_out.groupby("station_id")[target_reg].shift(-H)
        df_out[f"y_cls_{H}"] = (df_out[f"y_reg_{H}"] >= 1).astype("float")
        return df_out

    train_df = add_future_targets(train_df)
    test_df  = add_future_targets(test_df)

    # Filtrar filas con label disponible
    tr = train_df.dropna(subset=[f"y_reg_{H}"]).copy()
    te = test_df.dropna(subset=[f"y_reg_{H}"]).copy()
    if tr.empty or te.empty:
        return None  # insuficiente data para este H

    X_tr, y_tr = tr[feats], tr[f"y_reg_{H}"]
    X_te, y_te = te[feats], te[f"y_reg_{H}"]

    # ===== Modelos según config (sklearn_gbdt | lgbm) =====
    reg, cls_base, family = make_models(cfg)

    # Silenciar logs de LightGBM si aplica
    if family == "lgbm":
        # LightGBM acepta verbosity/verbose=-1
        for mdl in (reg, cls_base):
            for p in ("verbosity", "verbose"):
                try:
                    mdl.set_params(**{p: -1})
                except Exception:
                    pass
    else:
        # sklearn GradientBoosting: usar verbose=0 (o False)
        for mdl in (reg, cls_base):
            try:
                mdl.set_params(verbose=0)
            except Exception:
                pass


    # ===== Regressor =====
    reg.fit(X_tr, y_tr, sample_weight=(sample_w[:len(X_tr)] if sample_w is not None else None))
    yhat = reg.predict(X_te)
    mae = mean_absolute_error(y_te, yhat)

    # ===== Classifier (calibrado si se puede) =====
    ycls_tr = tr[f"y_cls_{H}"].astype(int)
    has_pos = (ycls_tr == 1).any()
    has_neg = (ycls_tr == 0).any()

    # Mejor calibración posible según datos
    method = "isotonic" if has_pos and has_neg and len(tr) >= 200 else "sigmoid"

    from sklearn.calibration import CalibratedClassifierCV
    import sklearn
    skl_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    calib_kwargs = {"cv": 3, "method": method}

    if has_pos and has_neg:
        # calibrado
        if skl_ver >= (1, 4):
            cal = CalibratedClassifierCV(estimator=cls_base, **calib_kwargs)
        else:
            cal = CalibratedClassifierCV(base_estimator=cls_base, **calib_kwargs)
        cal.fit(X_tr, ycls_tr, sample_weight=(sample_w[:len(X_tr)] if sample_w is not None else None))
        clf_for_pred = cal
    else:
        # sin ambas clases: entrenar sin calibrar
        cls_base.fit(X_tr, ycls_tr, sample_weight=(sample_w[:len(X_tr)] if sample_w is not None else None))
        clf_for_pred = cls_base

    # Probabilidades en test
    if hasattr(clf_for_pred, "predict_proba"):
        p = clf_for_pred.predict_proba(X_te)[:, 1]
    else:
        scores = clf_for_pred.decision_function(X_te)
        p = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    pr_auc = average_precision_score(te[f"y_cls_{H}"], p)

    # ===== Persistencia por horizonte =====
    hdir = REG_ROOT / f"h{H}"
    hdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, hdir / "regressor.joblib")
    joblib.dump(clf_for_pred, hdir / "classifier.joblib")
    joblib.dump({"features": feats, "features_version": "f1.0.0", "horizon_min": H}, hdir / "meta.joblib")

    # MLflow opcional
    if mlflow_active:
        try:
            import mlflow
            mlflow.log_metric(f"mae_h{H}", float(mae))
            mlflow.log_metric(f"pr_auc_h{H}", float(pr_auc))
        except Exception:
            pass

    return {
        "H": int(H),
        "mae": float(mae),
        "pr_auc": float(pr_auc),
        "family": family,
        "model_dir": str(hdir)
    }


def train(cfg):
    silver_pq = Path(cfg["data"]["base_path"]) / "silver" / "status_clean.parquet"
    df = pd.read_parquet(silver_pq)
    df.columns = [c.lower() for c in df.columns]

    required = ["ts_local", "station_id", cfg["training"]["target_reg"]]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Faltan columnas requeridas en silver: {missing}"

    df = ensure_datetime_local(df)
    if df.empty:
        raise RuntimeError("No hay registros con ts_local válido en silver. Revisa ETL/JSONs.")

    # Build features una sola vez
    df = build_features(df, cfg)
    if df.empty:
        raise RuntimeError("Tras generar features, no quedan filas (¿lags muy grandes vs data corto?).")

    # Split temporal (últimos N días a test). Si queda vacío, fallback 80/20
    last_ts = df["ts_local"].max()
    test_start = (last_ts.normalize() - pd.Timedelta(days=cfg["training"]["test_days"]))
    train_df = df[df["ts_local"] < test_start].copy()
    test_df  = df[df["ts_local"] >= test_start].copy()
    if train_df.empty or test_df.empty:
        split_idx = int(len(df)*0.8)
        train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    # Selección de features numéricas
    df, feats = pick_feats(df)
    train_df = train_df[feats + ["station_id","ts_local", cfg["training"]["target_reg"]]].copy()
    test_df  = test_df[feats + ["station_id","ts_local", cfg["training"]["target_reg"]]].copy()

    # Pesos por hora (opcional)
    hour_weights_path = Path("models/registry/hour_weights.json")
    sample_w = None
    if hour_weights_path.exists():
        wmap = json.loads(hour_weights_path.read_text())
        tr_hours = pd.to_datetime(train_df["ts_local"]).dt.hour.astype(str)
        sample_w = tr_hours.map(wmap).astype(float).values

    # MLflow (opcional)
    ml_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    mlflow_active = bool(ml_uri)
    if mlflow_active:
        mlflow.set_tracking_uri(ml_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "haybici-exp"))
        run = mlflow.start_run(run_name="multiH-gbdt")
        mlflow.log_params({
            "lags": cfg["features"]["lags_min"],
            "rolling": cfg["features"]["rolling_windows_min"]
        })

    results = []
    horizons = cfg["time"]["horizons_min"]
    for H in horizons:
        res = train_one_horizon(train_df.copy(), test_df.copy(), feats, H, cfg, sample_w=sample_w, mlflow_active=mlflow_active)
        if res is not None:
            results.append(res)

    # Índice de modelos disponibles
    (REG_ROOT / "index.json").write_text(json.dumps({
        "horizons_trained": [r["H"] for r in results],
        "features_version": "f1.0.0",
        "model_family": "gbdt"
    }, indent=2))

    if mlflow_active:
        for r in results:
            mlflow.log_metric(f"mae_h{r['H']}", r["mae"])
            mlflow.log_metric(f"prauc_h{r['H']}", r["pr_auc"])
        mlflow.end_run()

    # ===== Resumen limpio al final =====
    if not results:
        raise RuntimeError("No se pudo entrenar ningún horizonte (insuficiente histórico).")

    # Ordenar por H y guardar métricas en CSV/JSON
    results = sorted(results, key=lambda r: r["H"])
    
    df_res = pd.DataFrame(results)[["H","mae","pr_auc","family"]]
    metrics_csv = REG_ROOT / "metrics_horizons.csv"
    metrics_json = REG_ROOT / "metrics_horizons.json"
    df_res.to_csv(metrics_csv, index=False)
    metrics_json.write_text(_json.dumps(results, indent=2), encoding="utf-8")

    # Print final compacto
    print("\n===== Resumen por horizonte (final) =====")
    for r in results:
        print(f"H={r['H']:>3} min | MAE={r['mae']:.3f} | PR-AUC={r['pr_auc']:.3f} | family={r['family']}")
    print(f"\n📝 Guardado: {metrics_csv}")
    print(f"📝 Guardado: {metrics_json}")
    print("✅ Entrenamiento multi-horizonte completado.")


if __name__ == "__main__":
    cfg = load_cfg()
    train(cfg)
