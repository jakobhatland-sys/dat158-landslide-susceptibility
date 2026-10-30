"""
Landslide susceptibility pipeline (Voss/Granvin).
Input:
    data/raw/dtm10/dtm10_granvin.tif          (fra laz_to_dtm.py)
    data/processed/skred_jord_flom.geojson    (fra explore_skred.py)
Output:
    data/processed/dataset.csv
    models/model.joblib, models/metrics.json, models/features.tif
    models/susceptibility.tif
    report/figures/susceptibility_map.png, report/figures/feature_importance.png
Kjør fra prosjektroten:  python pipeline.py
"""
import json
import os

import geopandas as gpd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DTM = "data/raw/dtm10/6703_2_10m_z32.tif"
SKRED = "data/processed/skred_jord_flom.geojson"
MODEL_DIR, FIG_DIR, PROC_DIR = "models", "report/figures", "data/processed"
NEG_RATIO = 3       # pseudo-absence per positiv
BUFFER_M = 200      # minsteavstand negativ -> skred
BLOCK_M = 2000      # blokkstørrelse for spatial CV
SEED = 42
MIN_ELEV = 1.0      # maskerer fjord/sjø (høyde < 1 m)
rng = np.random.default_rng(SEED)
for d in (MODEL_DIR, FIG_DIR, PROC_DIR):
    os.makedirs(d, exist_ok=True)


def nan_mean_filter(a, size):
    m = np.isfinite(a)
    num = ndimage.uniform_filter(np.where(m, a, 0.0), size)
    den = ndimage.uniform_filter(m.astype("float32"), size)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


# ---------- 1. Terrengfeatures ----------
with rasterio.open(DTM) as src:
    z = src.read(1).astype("float32")
    T, crs, prof = src.transform, src.crs, src.profile
    if src.nodata is not None:
        z[z == src.nodata] = np.nan
z[z < MIN_ELEV] = np.nan
res = T.a
H, W = z.shape

g_row, g_col = np.gradient(z, res)
dzdx, dzdn = g_col, -g_row                      # øst, nord
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
aspect = np.arctan2(-dzdx, -dzdn)               # azimut for fallretning
curv = np.gradient(dzdx, res, axis=1) - np.gradient(dzdn, res, axis=0)  # Laplace, + = konkav
mean3 = nan_mean_filter(z, 3)
rough = np.sqrt(nan_mean_filter((z - mean3) ** 2, 3))   # lokal std, numerisk stabil i float32
tpi = z - nan_mean_filter(z, 15)                # topographic position index, 150 m

features = {
    "elevation": z, "slope": slope,
    "northness": np.cos(aspect), "eastness": np.sin(aspect),
    "curvature": curv, "roughness": rough, "tpi": tpi,
}
FEATS = list(features)
stack = np.stack([features[k] for k in FEATS]).astype("float32")
del features, g_row, g_col, dzdx, dzdn, aspect, curv, mean3, rough, tpi, slope  # frigjør minne
valid = np.all(np.isfinite(stack), axis=0)
print(f"DTM {W}x{H} @ {res} m, {valid.mean() * 100:.1f} % gyldige celler")

# ---------- 2. Positive og pseudo-absence ----------
skred = gpd.read_file(SKRED).to_crs(crs)
pts = skred.geometry.representative_point()
r, c = rowcol(T, pts.x.values, pts.y.values)
r, c = np.asarray(r), np.asarray(c)
inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
r, c = r[inside], c[inside]
keep = valid[r, c]
pos = np.unique(np.stack([r[keep], c[keep]], axis=1), axis=0)
print(f"Skred i fil: {len(skred)}, innenfor DTM og gyldige: {len(pos)}")
if len(pos) < 20:
    print("ADVARSEL: svært få positive. Resultatene blir usikre. Nevn dette i README.")

pmask = np.zeros_like(valid)
pmask[pos[:, 0], pos[:, 1]] = True
rad = int(round(BUFFER_M / res))
yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
excl = ndimage.binary_dilation(pmask, structure=(xx ** 2 + yy ** 2) <= rad ** 2)
cand = np.argwhere(valid & ~excl)
n_neg = min(len(cand), NEG_RATIO * len(pos))
neg = cand[rng.choice(len(cand), n_neg, replace=False)]

rc = np.vstack([pos, neg])
y = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)
X = stack[:, rc[:, 0], rc[:, 1]].T
xs, ys = xy(T, rc[:, 0], rc[:, 1])
xs, ys = np.asarray(xs), np.asarray(ys)
bx, by = np.floor(xs / BLOCK_M).astype(int), np.floor(ys / BLOCK_M).astype(int)
groups = pd.factorize(pd.Series(bx.astype(str)) + "_" + pd.Series(by.astype(str)))[0]
n_groups = len(np.unique(groups))
k = min(5, n_groups)
if k < 2:
    raise SystemExit("For få romlige blokker for spatial CV. Reduser BLOCK_M.")

df = pd.DataFrame(X, columns=FEATS)
df["label"], df["x"], df["y"], df["block"] = y, xs, ys, groups
df.to_csv(f"{PROC_DIR}/dataset.csv", index=False)
print(f"Datasett: {len(pos)} positive, {len(neg)} negative, {n_groups} blokker, {k}-fold")

# ---------- 3. Modeller og evaluering ----------
models = {
    "logistic_regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        n_jobs=-1, random_state=SEED),
}
results = {}
for name, m in models.items():
    p = cross_val_predict(m, X, y, cv=GroupKFold(n_splits=k), groups=groups,
                          method="predict_proba")[:, 1]
    results[name] = {"spatial_cv_roc_auc": round(roc_auc_score(y, p), 3),
                     "spatial_cv_pr_auc": round(average_precision_score(y, p), 3)}

# Random CV for å vise effekten av spatial autocorrelation
p_rand = cross_val_predict(models["random_forest"], X, y,
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                           method="predict_proba")[:, 1]
results["random_forest"]["random_cv_roc_auc"] = round(roc_auc_score(y, p_rand), 3)

best = max(results, key=lambda n: results[n]["spatial_cv_roc_auc"])
final = models[best].fit(X, y)
joblib.dump({"model": final, "features": FEATS, "name": best}, f"{MODEL_DIR}/model.joblib")

metrics = {"best_model": best, "models": results, "n_positive": int(len(pos)),
           "n_negative": int(len(neg)), "n_blocks": int(n_groups), "folds": int(k),
           "block_size_m": BLOCK_M, "features": FEATS, "crs": str(crs)}
with open(f"{MODEL_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(results, indent=2))
print(f"Beste modell (spatial ROC-AUC): {best}")

# ---------- 4. Feature importance ----------
rf = final if best == "random_forest" else models["random_forest"].fit(X, y)
order = np.argsort(rf.feature_importances_)
plt.figure(figsize=(6, 4))
plt.barh(np.array(FEATS)[order], rf.feature_importances_[order])
plt.xlabel("Mean decrease in impurity")
plt.title("Random forest feature importance")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=150)
plt.close()

# ---------- 5. Susceptibility-kart og feature-raster ----------
prob = np.full((H, W), np.nan, dtype="float32")
idx = np.argwhere(valid)
for i in range(0, len(idx), 200_000):
    b = idx[i:i + 200_000]
    prob[b[:, 0], b[:, 1]] = final.predict_proba(stack[:, b[:, 0], b[:, 1]].T)[:, 1]

out = prof.copy()
out.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
with rasterio.open(f"{MODEL_DIR}/susceptibility.tif", "w", **out) as dst:
    dst.write(prob, 1)

fst = np.where(valid, stack, np.nan).astype("float32")
out.update(count=len(FEATS))
with rasterio.open(f"{MODEL_DIR}/features.tif", "w", **out) as dst:
    dst.write(fst)
    for i, n in enumerate(FEATS, start=1):
        dst.set_band_description(i, n)

extent = (T.c, T.c + W * res, T.f + H * T.e, T.f)
px, py = xy(T, pos[:, 0], pos[:, 1])
plt.figure(figsize=(8, 7))
plt.imshow(prob, extent=extent, cmap="RdYlGn_r", vmin=0, vmax=1)
plt.colorbar(label="Relative susceptibility score", shrink=0.8)
plt.scatter(px, py, s=8, c="k", label="Registered events (NVE)")
plt.legend(loc="lower right")
plt.title(f"Landslide susceptibility – {best}")
plt.xlabel(f"Easting ({crs})")
plt.ylabel("Northing")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/susceptibility_map.png", dpi=150)
plt.close()
print("Ferdig. Kjør appen: streamlit run app.py")
