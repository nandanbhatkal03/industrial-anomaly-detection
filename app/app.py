"""Streamlit inspection dashboard.   Run:  streamlit run app/app.py"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from app.components.result_view import render_result  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.utils.config import load_config  # noqa: E402

st.set_page_config(page_title="Industrial Defect Inspection", page_icon="🔍", layout="wide")
cfg = load_config()
CK_DIR, LOG = ROOT / cfg["paths"]["checkpoints"], ROOT / cfg["paths"]["results"] / "inspection_log.csv"


@st.cache_resource(show_spinner="Loading model...")
def get_predictor(path: str, device: str):
    return Predictor(path, device, cfg)


st.title("🔍 Lightweight Visual Anomaly Detection - Inspection Dashboard")
ckpts = sorted(CK_DIR.glob("*.pt"))
if not ckpts:
    st.warning("No checkpoints found in models/checkpoints. Train one first: "
               "`python scripts/train.py --method lightpatch --category bottle`")
    st.stop()

with st.sidebar:
    st.header("Model")
    ck = st.selectbox("Checkpoint", ckpts, format_func=lambda p: p.stem)
    device = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"])
    pred = get_predictor(str(ck), device)
    cal = pred.calibration
    st.caption(f"method: **{pred.meta['method']}** | backbone: **{pred.meta['model_cfg'].get('backbone', '-')}**")
    st.header("Decision")
    sens = st.slider("Threshold multiplier", 0.5, 1.5, 1.0, 0.05,
                     help="<1 = more sensitive (more alarms); >1 = stricter. 1.0 = calibrated on normal validation images.")
    post = st.checkbox("Morphological clean-up", True)
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.5, 0.05)
    st.caption(f"calibrated image thr = {cal['image_threshold']:.4f} | pixel thr = {cal['pixel_threshold']:.4f}")

files = st.file_uploader("Upload industrial image(s)", type=["png", "jpg", "jpeg", "bmp", "tif"], accept_multiple_files=True)
for f in files or []:
    st.subheader(f.name)
    r = pred.predict(Image.open(f), cal["image_threshold"] * sens, cal["pixel_threshold"] * sens, post)
    render_result(r, alpha)
    new = not LOG.exists()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "file", "model", "score", "threshold", "anomalous", "severity", "n_regions", "latency_ms"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), f.name, ck.stem, f"{r.score:.5f}",
                    f"{r.image_threshold:.5f}", r.is_anomalous, r.severity, len(r.regions), f"{r.timing_ms['total']:.1f}"])

st.divider()
run = ck.stem
m = ROOT / cfg["paths"]["results"] / "metrics" / f"{run}.json"
with st.expander("Offline evaluation summary for this model", expanded=False):
    if m.exists():
        row = json.loads(m.read_text())["row"]
        keys = ["image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_iou", "latency_ms_mean", "fps",
                "params_m", "deployed_size_mb"]
        st.table({k: [round(float(row[k]), 4)] for k in keys if k in row})
    else:
        st.info("No evaluation file yet. Run `python scripts/evaluate.py --checkpoint ... --category ...`")
