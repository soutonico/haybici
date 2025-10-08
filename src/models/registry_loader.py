from pathlib import Path
import joblib
import json
from functools import lru_cache

REG_ROOT = Path("models/registry")

def available_horizons():
    idx = REG_ROOT / "index.json"
    if idx.exists():
        try:
            data = json.loads(idx.read_text())
            return sorted([int(h) for h in data.get("horizons_trained", [])])
        except Exception:
            return []
    # Fallback: detectar carpetas hXX
    hs = []
    for p in REG_ROOT.glob("h*"):
        try:
            hs.append(int(p.name[1:]))
        except:
            pass
    return sorted(hs)

@lru_cache(maxsize=64)
def load_models_for_horizon(H: int):
    hdir = REG_ROOT / f"h{H}"
    reg = joblib.load(hdir / "regressor.joblib")
    cls = joblib.load(hdir / "classifier.joblib")
    meta = joblib.load(hdir / "meta.joblib")
    feats = meta["features"]
    fver = meta.get("features_version","f1.0.0")
    return reg, cls, feats, fver
