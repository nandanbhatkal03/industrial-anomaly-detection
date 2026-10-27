"""Optional REST API.   Run:  uvicorn app.api:app --port 8000
POST /predict?checkpoint=<name>&sensitivity=1.0  (multipart file field 'file')"""
import io
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from PIL import Image  # noqa: E402

from src.inference.predictor import Predictor  # noqa: E402
from src.utils.config import load_config  # noqa: E402

cfg = load_config()
CK_DIR = ROOT / cfg["paths"]["checkpoints"]
app = FastAPI(title="Industrial anomaly detection API", version="0.1")


@lru_cache(maxsize=4)
def _predictor(name: str) -> Predictor:
    path = CK_DIR / f"{name}.pt"
    if not path.is_file() or path.resolve().parent != CK_DIR.resolve():  # no path traversal
        raise HTTPException(404, f"checkpoint '{name}' not found")
    return Predictor(path, cfg.get("device", "auto"), cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return {"checkpoints": sorted(p.stem for p in CK_DIR.glob("*.pt"))}


@app.post("/predict")
async def predict(checkpoint: str, file: UploadFile = File(...), sensitivity: float = 1.0):
    p = _predictor(checkpoint)  # validates the name first (404)
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "could not decode image")
    cal = p.calibration
    r = p.predict(img, cal["image_threshold"] * sensitivity, cal["pixel_threshold"] * sensitivity)
    return r.to_dict()
