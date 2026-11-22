
import pandas as pd
from pathlib import Path

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

files = {
    "suppliers": RAW_DATA_DIR / "suppliers.csv",
    "inventory": RAW_DATA_DIR / "inventory.csv",
    "sales_history": RAW_DATA_DIR / "sales_history.csv",
    "warehouses": RAW_DATA_DIR / "warehouses.csv",
    "replenishment_v5": PROCESSED_DATA_DIR / "replenishment_requirements_v5.csv",
    "supplier_allocation_v5": PROCESSED_DATA_DIR / "supplier_allocation_v5.csv",
}

for name, file_path in files.items():
    print(f"\n{'=' * 55}")
    print(f"DATASET: {name}")
    print(f"PATH: {file_path}")

    if not file_path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))


from pathlib import Path

from config import BASE_DIR

folders = [
    BASE_DIR / "data",
    BASE_DIR / "data" / "processed",
    BASE_DIR / "reports",
]

keywords = [
    "assignment",
    "replenishment",
    "supplier_allocation",
    "optimized_routes",
    "logistics_summary",
]

for folder in folders:
    print(f"\n{'=' * 55}")
    print("FOLDER:", folder)

    if not folder.exists():
        print("Folder does not exist.")
        continue

    for file_path in folder.rglob("*.csv"):
        if any(word in file_path.name.lower() for word in keywords):
            print(file_path)

import pandas as pd

from config import REPORT_DIR

files = {
    "store_warehouse_assignment": REPORT_DIR / "store_warehouse_assignment.csv",
    "replenishment_v5": REPORT_DIR / "replenishment_requirements_v5.csv",
    "supplier_allocation_v5": REPORT_DIR / "supplier_allocation_v5.csv",
}

for name, file_path in files.items():
    print(f"\n{'=' * 60}")
    print("DATASET:", name)

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
