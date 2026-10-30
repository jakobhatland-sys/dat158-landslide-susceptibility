"""
Streamlit app – landslide susceptibility, Voss/Granvin.
Kjør fra prosjektroten:  streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from rasterio.warp import transform as warp_transform

st.set_page_config(page_title="Landslide susceptibility – Voss", layout="wide")

CRS = "EPSG:25832"   # rasterets ESRI-CRS tolkes ikke korrekt av PROJ, så det angis eksplisitt


@st.cache_resource
def load():
    bundle = joblib.load("models/model.joblib")
    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with rasterio.open("models/features.tif") as src:
        stack, T, bounds = src.read(), src.transform, src.bounds
    return bundle, metrics, stack, T, bounds


bundle, metrics, stack, T, bounds = load()
model, FEATS = bundle["model"], bundle["features"]
_, H, W = stack.shape

st.title("Landslide susceptibility – Voss / Granvin")
st.caption("Soil slides and debris flows (jord- og flomskred). "
           "Model trained on NVE landslide events and terrain derivatives from a 10 m DTM.")

cx, cy = (bounds.left + bounds.right) / 2, (bounds.bottom + bounds.top) / 2
lon0, lat0 = warp_transform(CRS, "EPSG:4326", [cx], [cy])

left, right = st.columns(2)
with left:
    st.subheader("Query a location")
    lat = st.number_input("Latitude (WGS84)", value=float(lat0[0]), format="%.5f")
    lon = st.number_input("Longitude (WGS84)", value=float(lon0[0]), format="%.5f")
    x, y = warp_transform("EPSG:4326", CRS, [lon], [lat])
    col = int((x[0] - T.c) / T.a)
    row = int((y[0] - T.f) / T.e)

    if not (0 <= row < H and 0 <= col < W):
        st.error("Location is outside the study area.")
    else:
        vals = stack[:, row, col]
        if not np.all(np.isfinite(vals)):
            st.warning("No terrain data at this location (e.g. water or data gap).")
        else:
            p = float(model.predict_proba(vals.reshape(1, -1))[0, 1])
            level = "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"
            st.metric("Relative susceptibility score", f"{p:.2f}", level)
            st.dataframe(pd.DataFrame({"feature": FEATS, "value": np.round(vals, 2)}),
                         hide_index=True)
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=11)

with right:
    st.subheader("Susceptibility map")
    st.image("report/figures/susceptibility_map.png")

st.subheader("Model evaluation (spatial block cross-validation)")
st.dataframe(pd.DataFrame(metrics["models"]).T)
st.write(f"Best model: **{metrics['best_model']}** · "
         f"{metrics['n_positive']} events, {metrics['n_negative']} pseudo-absence points, "
         f"{metrics['folds']}-fold spatial CV over {metrics['n_blocks']} blocks "
         f"of {metrics['block_size_m']} m.")
st.image("report/figures/feature_importance.png", width=500)

st.info("The score is a relative susceptibility index, not a probability of occurrence: "
        "negatives are pseudo-absence points and the class ratio affects calibration. "
        "Terrain features only (no precipitation, soil or vegetation). "
        "Not a substitute for NVE's official hazard maps.")