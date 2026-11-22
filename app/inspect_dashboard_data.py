
from pathlib import Path
import pandas as pd

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data" / "raw"

files_to_inspect = {
    "Products": DATA_DIR / "products.csv",
    "Suppliers": DATA_DIR / "suppliers.csv",
    "Warehouses": DATA_DIR / "warehouses.csv",
    "Stores": DATA_DIR / "stores.csv",
    "Sales History": DATA_DIR / "sales_history.csv",
    "Forecasting Results": REPORT_DIR / "forecasting_results.csv",
    "Procurement Summary": REPORT_DIR / "optimization_summary_v5.csv",
    "Supplier Summary": REPORT_DIR / "supplier_summary_v5.csv",
    "Simulation Comparison": REPORT_DIR / "simulation_scenario_comparison.csv",
    "AI Recommendations": REPORT_DIR / "llm_agent_recommendations_critic_reviewed.csv",
}

for name, file_path in files_to_inspect.items():
    print("\n" + "=" * 65)
    print(f"{name}")
    print("=" * 65)

    if not file_path.exists():
        print(f"FILE NOT FOUND: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nFirst 2 rows:")
    print(df.head(2).to_string(index=False))
