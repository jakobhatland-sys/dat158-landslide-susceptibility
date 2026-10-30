"""Central configuration: paths and parameters shared by all pipeline steps."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# --- paths -----------------------------------------------------------------
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "report" / "figures"

EVENTS_RAW_DIR = RAW / "skredhendelser"
DTM_RAW_DIR = RAW / "dtm10"

EVENTS_CLEAN = PROCESSED / "events_clean.gpkg"
DTM_MOSAIC = PROCESSED / "dtm.tif"
FEATURE_DIR = PROCESSED / "features"
DATASET = PROCESSED / "dataset.parquet"

for p in (PROCESSED, MODELS, FIGURES, FEATURE_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- parameters ------------------------------------------------------------
CRS = "EPSG:25833"          # EUREF89 / UTM zone 33N
RESOLUTION = 25             # m – working resolution (DTM10 resampled; keeps Vestland tractable)
RANDOM_STATE = 42

# Events: max accepted position uncertainty (m). Set after EDA of the accuracy field.
MAX_POS_UNCERTAINTY = 100

# Negative sampling
NEG_POS_RATIO = 3           # negatives per positive
NEG_BUFFER_M = 500          # min distance from any event

# Spatial cross-validation
BLOCK_SIZE_M = 10_000       # 10 km blocks
N_FOLDS = 5
