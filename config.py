
from pathlib import Path

# Main project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Other directories
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

# Simulation configuration
NUM_SUPPLIERS = 3
NUM_WAREHOUSES = 10
NUM_STORES = 20
NUM_PRODUCTS = 100
HISTORICAL_DAYS = 180

# Random seed for reproducibility
RANDOM_SEED = 42

# Create required directories
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    REPORT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

print("Project configuration loaded successfully.")
