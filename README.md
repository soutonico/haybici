# HayBici?

Predice disponibilidad de bicicletas en **CABA (EcoBici)** por estación y **minuto futuro**.  
Devuelve:
- **`y_hat`**: cantidad esperada de bicis.
- **`p_availability`**: probabilidad **calibrada** de que haya ≥1 bici.

Modelos **multi-horizonte** (H en minutos): 1,3,5,10,15,20,25…120 (configurable).

---

## Pasos rápidos

1) **Entorno**
```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


2) **Configurar**

    - Copiá .env.example → .env (si usás MLflow/MinIO).

    - Revisá config/config.yaml (zona horaria, features, horizontes, familia de modelo, pesos por hora).

3) **Ingesta → bronze/silver**

  ```bash
  python scripts\make_dataset.py `
    --info-json-glob "data/bronze/raw/stationInformation_*.json" `
    --status-json-glob "data/bronze/raw/stationStatus_*.json" `
    --status-parquet "C:\ruta\ecobici_station_status.parquet"    # opcional
  ```

4) **Pesos por hora**
  ```bash
  python scripts/update_hourly_weights.py
  ```

5) **Entrenaimento multihorizonte**
  ```bash
  python scripts/train_model.py
  ```

- Modelos por cada H en config.yaml.

- Guarda artefactos en models/registry/h{H}/.

- Resumen final:

  - models/registry/metrics_horizons.csv

  - models/registry/metrics_horizons.json

6) **Servir API**
```bash
bash scripts/serve_api.sh
# o
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
```

7) **Probar**
  ```bash
GET http://localhost:8080/predict?lat=-34.6037&lon=-58.3816&minutosLlegada=17
GET http://localhost:8080/predict?lat=-34.6037&lon=-58.3816&horaLlegada=14:35
 ```

## Estructura
  ```bash
HayBici/
├─ config/
│  └─ config.yaml
├─ data/
│  ├─ bronze/
│  │  ├─ raw/                         # (opcional) JSON GBFS originales
│  │  ├─ station_information.parquet
│  │  └─ station_status.parquet
│  └─ silver/
│     └─ status_clean.parquet         # dataset limpio (hora local)
├─ models/
│  └─ registry/
│     ├─ h1/, h3/, h5/, ...           # por horizonte
│     │  ├─ regressor.joblib
│     │  ├─ classifier.joblib         # calibrado si aplica
│     │  └─ meta.joblib
│     ├─ index.json                   # horizontes entrenados
│     ├─ hour_weights.json            # (opcional) pesos por hora
│     ├─ metrics_horizons.csv
│     └─ metrics_horizons.json
├─ scripts/
│  ├─ make_dataset.py
│  ├─ train_model.py
│  └─ update_hourly_weights.py
└─ src/
   └─ api/
      └─ main.py
```

## ¿Qué hace cada script?

### scripts/make_dataset.py

- Lee múltiples JSON GBFS (station_information / station_status) por glob y/o un Parquet unido (opcional).

- Escribe bronze (sanitiza dicts/listas conflictivas para Parquet).

- Genera silver:

  - ts_local robusto (de last_reported con fallback _file_last_updated, en BA).

  - Join con information (name, lat, lon, capacity).

  - Filtra estaciones no operativas (is_renting/is_returning).

  - Ordena por station_id, ts_local.

### scripts/update_hourly_weights.py

- Aproxima salidas por minuto:
  - delta_bikes = diff(bikes) → dep_estimada = max(-delta, 0).

- Robustez configurable (en config.yaml > weights):

  - Micro-ruido (noise_threshold).

  - Cap por capacity.

  - Outliers imposibles (max_delta_per_min).

- Agrega por hora local, normaliza a [min_weight, max_weight], guarda hour_weights.json.

### scripts/train_model.py

- Features (sin leakage): lags, rolling mean/std (con shift(1)), estacionalidad semanal (one-hot) y horaria (sin/cos).

- Split temporal (últimos test_days para test; fallback 80/20 si el histórico es corto).

- Targets a futuro por H:

  - y_reg_H(t) = bikes(t+H),

  - y_cls_H(t) = 1[y_reg_H≥1].

- Entrena por H:

  - Regresor (cantidad).

  - Clasificador calibrado (probabilidad) con CalibratedClassifierCV (isotónica si hay datos y ambas clases, si no sigmoide; si falta clase, sin calibrar).

  - Usa sample_weight por hora si existe hour_weights.json.

- Persiste artefactos en models/registry/h{H}/ y crea index.json.

- Resumen limpio final (sin spam) + métricas a CSV/JSON.

## Configuración (config/config.yaml)

  ```bash
time:
  local_tz: "America/Argentina/Buenos_Aires"
  resolution_min: 1
  max_horizon_min: 120
  horizons_min: [1, 3, 5, 10, 15, 20, 25, 30, 60, 120]

training:
  target_reg: "num_bikes_available"
  test_days: 7
  target_cls_threshold: 1

features:
  lags_min: [1, 3, 5, 10, 15, 20]
  rolling_windows_min: [5, 15, 30]
  add_weekly_seasonality: true
  add_hourly_cyclical: true

weights:
  noise_threshold: 0.5   # contar caídas de 1 bici (sí)
  cap_by_capacity: true
  max_delta_per_min: 30
  min_weight: 0.5
  max_weight: 1.5

model:
  family: "sklearn_gbdt"   # opciones: sklearn_gbdt | lgbm

  sklearn_gbdt:
    regressor:  { random_state: 42 }
    classifier: { random_state: 42 }

  lgbm:
    regressor:
      n_estimators: 1000
      learning_rate: 0.05
      num_leaves: 63
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
    classifier:
      n_estimators: 1000
      learning_rate: 0.05
      num_leaves: 63
      subsample: 0.8
      colsample_bytree: 0.8
      random_state: 42
```

## API (contrato)

GET /predict?lat=…&lon=…&minutosLlegada=…

GET /predict?lat=…&lon=…&horaLlegada=HH:mm

Respuesta (top-k estaciones; k configurable, default 3):

```json
[
  {
    "station_id": "123",
    "distance_m": 210.5,
    "y_hat": 4.7,
    "p_availability": 0.92,
    "prediction_ts": "2025-10-03T14:35:00-03:00",
    "horizon_min": 17,
    "features_version": "f1.0.0",
    "model_version": "m_gbdt_h15"   // H del modelo elegido (más cercano al pedido)
  }
]

// Si pedís 17 min y existen modelos {15, 20}, se usa el más cercano (15).
// Sin modelos entrenados: fallback simple (estado actual).
```

## Métricas

- MAE (regresión): error medio en bicicletas (↓ mejor). Sube con H (más incertidumbre).

- PR-AUC (clasificación): calidad para distinguir ≥1 bici vs 0 con desbalance (↑ mejor).

- Baseline recomendado: persistencia (usar lag_H como predicción) para medir “lift”.

