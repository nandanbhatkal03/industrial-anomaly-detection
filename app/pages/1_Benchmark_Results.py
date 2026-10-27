import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Benchmark results", layout="wide")
st.title("Benchmark results")
csv_path = ROOT / "results" / "tables" / "summary.csv"
if not csv_path.exists():
    st.info("Run `python scripts/run_experiments.py` first.")
    st.stop()
df = pd.read_csv(csv_path)
cols = ["category", "method", "tag", "image_auroc", "pixel_auroc", "aupro", "image_f1", "pixel_iou",
        "latency_ms_mean", "fps", "params_m", "deployed_size_mb"]
st.dataframe(df[[c for c in cols if c in df]])
agg = df.groupby(["method", "tag"])[[c for c in cols[3:] if c in df]].mean().reset_index()
st.subheader("Mean over categories")
st.dataframe(agg)
st.scatter_chart(agg, x="latency_ms_mean", y="image_auroc", color="method")
fig = ROOT / "results" / "visualizations" / "tradeoff.png"
if fig.exists():
    st.image(str(fig))
