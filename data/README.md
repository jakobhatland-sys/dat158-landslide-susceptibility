# Data

All raw data goes in `data/raw/`. Nothing in `data/` is committed except this file.

## 1. NVE Skredhendelser (landslide events)

- Portal: https://nedlasting.nve.no/gis/
- Dataset: **Skred → Skredhendelser**
- Area: **Fylke → Vestland**
- Format: **Geopackage** if offered, otherwise **ESRI Shape** or FGDB
- Projection: **EUREF89 UTM sone 33 (EPSG:25833)**
- NVE emails a download link. Unzip into `data/raw/skredhendelser/`.
- Download date: _YYYY-MM-DD_ (record this – the dataset is continuously updated)

## 2. Kartverket DTM 10 (UTM33)

- Portal: https://kartkatalog.geonorge.no → search "DTM 10 Terrengmodell (UTM33)"
- Area: **Fylke → Vestland**
- Format: **GeoTIFF**
- Projection: **EUREF89 UTM sone 33 (EPSG:25833)**
- Unzip into `data/raw/dtm10/` (one or more `.tif` tiles)
- Download date: _YYYY-MM-DD_
