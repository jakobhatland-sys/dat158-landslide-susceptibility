"""
Hent NVE skredhendelser (utløsningspunkt) for Voss/Granvin direkte fra NVEs
ArcGIS REST-tjeneste, dekod skredtype og filtrer til jord- og flomskred.
Ingen bestilling nødvendig.

Kjør fra prosjektroten:  python fetch_skred.py
Output:
    data/raw/skredhendelser_voss.geojson       (alle skredtyper)
    data/processed/skred_jord_flom.geojson     (filtrert, brukes av pipeline.py)
"""
import os
import requests
import geopandas as gpd

SERVICES = [
    "https://kart.nve.no/enterprise/rest/services/Skredhendelser1/MapServer/1",
    "https://gis3.nve.no/map/rest/services/Mapservices/SkredHendelser/MapServer/1",
]
BBOX_WGS84 = "5.9,60.35,7.1,60.95"     # dekker DTM-flis 6703_2
RAW = "data/raw/skredhendelser_voss.geojson"
OUT = "data/processed/skred_jord_flom.geojson"
PAGE = 1000


def layer_info():
    for url in SERVICES:
        try:
            info = requests.get(url, params={"f": "json"}, timeout=30).json()
            if "fields" in info:
                print("Bruker:", url)
                return url, info
        except Exception as e:
            print("Feilet:", url, "-", e)
    raise SystemExit("Ingen tjeneste svarte. Prøv igjen eller bruk nettleser-URL i README.")


def code_labels(info, field):
    """Kodeliste for skredtype: domene først, ellers renderer."""
    for f in info["fields"]:
        if f["name"].lower() == field.lower() and f.get("domain"):
            return {int(c["code"]): c["name"] for c in f["domain"].get("codedValues", [])}, "domain"
    r = info.get("drawingInfo", {}).get("renderer", {})
    vals = r.get("uniqueValueInfos", [])
    return {int(v["value"]): v["label"] for v in vals if str(v["value"]).lstrip("-").isdigit()}, "renderer"


def fetch(url):
    feats, offset = [], 0
    while True:
        params = {
            "where": "1=1", "geometry": BBOX_WGS84,
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
            "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": PAGE,
        }
        batch = requests.get(f"{url}/query", params=params, timeout=60).json().get("features", [])
        feats += batch
        print(f"  hentet {len(feats)}")
        if len(batch) < PAGE:
            return feats
        offset += PAGE


url, info = layer_info()
type_field = next((f["name"] for f in info["fields"] if f["name"].lower() == "skredtype"), None)
if type_field is None:
    raise SystemExit(f"Fant ikke skredType-felt. Felter: {[f['name'] for f in info['fields']]}")

labels, source = code_labels(info, type_field)
feats = fetch(url)
if not feats:
    raise SystemExit("Ingen objekter returnert.")

gdf = gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")
gdf["skredtype_navn"] = gdf[type_field].map(labels).fillna("ukjent")
os.makedirs(os.path.dirname(RAW), exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
gdf.to_file(RAW, driver="GeoJSON")

print(f"\nAlle hendelser i bbox: {len(gdf)}  (kodeliste fra {source})")
print(gdf.groupby([type_field, "skredtype_navn"]).size().to_string())

name = gdf["skredtype_navn"].str.lower()
mask = name.str.contains("jord|flom")
if source == "domain":
    mask &= ~name.str.contains("kvikk")
else:
    print("\nMERK: kodeliste fra renderer, kvikkleire kan være slått sammen med jordskred.")

sel = gdf[mask]
sel.to_file(OUT, driver="GeoJSON")
print(f"\nJord-/flomskred: {len(sel)}  ->  {OUT}")
print(sel["skredtype_navn"].value_counts().to_string())
