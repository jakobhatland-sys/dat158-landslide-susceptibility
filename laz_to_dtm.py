"""
Lag DTM (10 m GeoTIFF) fra LAZ-punktsky uten PDAL.
- Kun bakkepunkter (ASPRS-klasse 2)
- Middelverdi per 10 m celle
- Reprojiserer fra EPSG:25832 (UTM32) til EPSG:25833 (UTM33)

Installasjon:
    pip install "laspy[lazrs]" numpy rasterio
Kjør fra prosjektroten:
    python src/laz_to_dtm.py
"""
import glob
import os
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

LAZ_DIR = r"D:\granvin\terrengmodell\gemini_ferdigzippet\3765\data"
OUT = "data/raw/dtm10/dtm10_granvin.tif"
TMP = "data/raw/dtm10/_dtm10_utm32.tif"
RES = 10.0
SRC_CRS = "EPSG:25832"
DST_CRS = "EPSG:25833"
NODATA = -9999.0

files = sorted(glob.glob(os.path.join(LAZ_DIR, "*.laz")))
if not files:
    raise SystemExit(f"Ingen LAZ-filer i {LAZ_DIR}")
print(f"{len(files)} LAZ-filer funnet")

# 1) Felles utstrekning fra headerne (rask, leser ikke punktene)
xmin = ymin = np.inf
xmax = ymax = -np.inf
for f in files:
    with laspy.open(f) as r:
        h = r.header
        xmin, ymin = min(xmin, h.mins[0]), min(ymin, h.mins[1])
        xmax, ymax = max(xmax, h.maxs[0]), max(ymax, h.maxs[1])

xmin, ymax = np.floor(xmin / RES) * RES, np.ceil(ymax / RES) * RES
ncols = int(np.ceil((xmax - xmin) / RES))
nrows = int(np.ceil((ymax - ymin) / RES))
print(f"Rutenett: {ncols} x {nrows} celler")

zsum = np.zeros(nrows * ncols, dtype=np.float64)
zcnt = np.zeros(nrows * ncols, dtype=np.int64)

# 2) Akkumuler bakkepunkter fil for fil (lavt minnebruk)
total = 0
for f in files:
    las = laspy.read(f)
    g = np.asarray(las.classification) == 2
    x, y, z = np.asarray(las.x)[g], np.asarray(las.y)[g], np.asarray(las.z)[g]
    col = ((x - xmin) / RES).astype(np.int64)
    row = ((ymax - y) / RES).astype(np.int64)
    ok = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    idx = row[ok] * ncols + col[ok]
    zsum += np.bincount(idx, weights=z[ok], minlength=zsum.size)
    zcnt += np.bincount(idx, minlength=zcnt.size)
    total += int(ok.sum())
    print(f"  {os.path.basename(f)}: {ok.sum():,} bakkepunkter")

if total == 0:
    raise SystemExit("0 bakkepunkter. Punktskyen er trolig ikke klassifisert.")

dtm = np.full(zsum.size, NODATA, dtype=np.float32)
has = zcnt > 0
dtm[has] = (zsum[has] / zcnt[has]).astype(np.float32)
dtm = dtm.reshape(nrows, ncols)
print(f"Totalt {total:,} bakkepunkter, {100 * has.mean():.1f} % celler med data")

# 3) Skriv i UTM32, deretter reprojiser til UTM33
os.makedirs(os.path.dirname(OUT), exist_ok=True)
profile = dict(driver="GTiff", height=nrows, width=ncols, count=1,
               dtype="float32", crs=SRC_CRS, nodata=NODATA,
               transform=from_origin(xmin, ymax, RES, RES), compress="deflate")
with rasterio.open(TMP, "w", **profile) as dst:
    dst.write(dtm, 1)

with rasterio.open(TMP) as src:
    transform, w, h = calculate_default_transform(
        src.crs, DST_CRS, src.width, src.height, *src.bounds, resolution=RES)
    out_prof = src.profile.copy()
    out_prof.update(crs=DST_CRS, transform=transform, width=w, height=h)
    with rasterio.open(OUT, "w", **out_prof) as dst:
        reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=DST_CRS,
                  src_nodata=NODATA, dst_nodata=NODATA,
                  resampling=Resampling.bilinear)

os.remove(TMP)
with rasterio.open(OUT) as r:
    print(f"\nLagret: {os.path.abspath(OUT)}")
    print(f"CRS: {r.crs}  Bounds: {tuple(round(b) for b in r.bounds)}")
