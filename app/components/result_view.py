import numpy as np
import streamlit as st

from src.inference.heatmap import draw_regions, overlay_heatmap, overlay_mask


def render_result(r, alpha: float = 0.5) -> None:
    badge = "🔴 ANOMALY" if r.is_anomalous else "🟢 NORMAL"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision", badge)
    c2.metric("Anomaly score", f"{r.score:.4f}", f"threshold {r.image_threshold:.4f}", delta_color="off")
    c3.metric("Severity", f"{r.severity} ({r.severity_index:.2f})")
    c4.metric("Latency", f"{r.timing_ms['total']:.0f} ms")
    t1, t2, t3 = st.tabs(["Heatmap", "Defect regions", "Details"])
    with t1:
        a, b = st.columns(2)
        a.image(r.image, caption="Input (as seen by the model)", width=380)
        b.image(overlay_heatmap(r.image, r.norm_map, alpha), caption="Anomaly heatmap (0.5 = threshold)",
                width=380)
    with t2:
        st.image(draw_regions(overlay_mask(r.image, r.mask), r.regions), width=380)
        if r.regions:
            st.dataframe([{"id": g["id"], "bbox (x,y,w,h)": g["bbox"], "area_px": g["area_px"],
                           "area_%": round(100 * g["area_ratio"], 2), "max_score": round(g["max_score"], 4)}
                          for g in r.regions])
        else:
            st.info("No defect region reported.")
    with t3:
        st.json(r.to_dict())
