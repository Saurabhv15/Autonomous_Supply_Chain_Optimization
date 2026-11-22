
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"

files = [
    "replenishment_requirements_v5.csv",
    "supplier_allocation_v5.csv",
    "procurement_shortage_v5.csv",
    "store_warehouse_assignment.csv",
    "optimized_routes.csv",
    "warehouse_logistics_summary.csv",
    "logistics_optimization_summary.csv"
]

for filename in files:

    file_path = REPORT_DIR / filename

    print("\n" + "=" * 75)
    print(f"FILE: {filename}")
    print("=" * 75)

    if not file_path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
